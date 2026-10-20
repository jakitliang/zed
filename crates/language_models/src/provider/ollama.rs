use anyhow::{Result, anyhow};
use collections::HashMap;
use credentials_provider::CredentialsProvider;
use fs::Fs;
use futures::{FutureExt, StreamExt, channel::mpsc, future::BoxFuture};
use futures::{Stream, TryFutureExt, stream};
use gpui::{App, AsyncApp, Context, Entity, Task, TaskExt};
use http_client::{CustomHeaders, HttpClient};
use language_model::{
    ApiKeyState, AuthenticateError, DisabledReason, EnvVar, IconOrSvg, InlineDescription,
    LanguageModel, LanguageModelClient, LanguageModelCompletionError, LanguageModelCompletionEvent,
    LanguageModelCompletionStream, LanguageModelId, LanguageModelName, LanguageModelProvider,
    LanguageModelProviderId, LanguageModelProviderName, LanguageModelProviderState,
    LanguageModelRequest, LanguageModelRequestTool, LanguageModelToolUse, LanguageModelToolUseId,
    MessageContent, ModelRateLimiters, ProviderSettingsView, Role, StopReason,
    SubPageProviderSettings, TokenUsage, env_var, unavailable_error,
};
use menu;
use ollama::{
    ChatMessage, ChatOptions, ChatRequest, ChatResponseDelta, OLLAMA_API_URL, OllamaFunctionCall,
    OllamaFunctionTool, OllamaToolCall, get_models, get_running_models, show_model,
    stream_chat_completion,
};
pub use settings::OllamaAvailableModel as AvailableModel;
use settings::{Settings, SettingsStore, update_settings_file};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex;
use std::sync::Weak;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use ui::{
    ButtonLike, ButtonLink, ConfiguredApiCard, Divider, List, ListBulletItem, Tooltip, prelude::*,
};
use ui_input::InputField;

use crate::AllLanguageModelSettings;

const OLLAMA_DOWNLOAD_URL: &str = "https://ollama.com/download";
const OLLAMA_LIBRARY_URL: &str = "https://ollama.com/library";
const OLLAMA_SITE: &str = "https://ollama.com/";

const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("ollama");
const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("Ollama");

const API_KEY_ENV_VAR_NAME: &str = "OLLAMA_API_KEY";
static API_KEY_ENV_VAR: LazyLock<EnvVar> = env_var!(API_KEY_ENV_VAR_NAME);
const RUNTIME_CONTEXT_TTL: Duration = Duration::from_secs(60);

#[derive(Default)]
struct CompletionCapacityObservations {
    active: Mutex<HashMap<(String, String), Vec<Weak<AtomicBool>>>>,
}

impl CompletionCapacityObservations {
    fn begin(&self, api_url: &str, model_name: &str) -> Arc<AtomicBool> {
        let mut active = self
            .active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let observations = active
            .entry((
                api_url.to_owned(),
                model_name
                    .strip_suffix(":latest")
                    .unwrap_or(model_name)
                    .to_owned(),
            ))
            .or_default();
        observations.retain(|observation| {
            if let Some(overlapped) = observation.upgrade() {
                // A completed competing request still invalidates the older observation.
                overlapped.store(true, Ordering::SeqCst);
                true
            } else {
                false
            }
        });
        let overlapped = Arc::new(AtomicBool::new(!observations.is_empty()));
        observations.push(Arc::downgrade(&overlapped));
        overlapped
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct OllamaSettings {
    pub api_url: String,
    pub auto_discover: bool,
    pub available_models: Vec<AvailableModel>,
    pub context_window: Option<u64>,
    pub custom_headers: CustomHeaders,
}

pub struct OllamaLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
    request_limiters: ModelRateLimiters,
    completion_notifications: mpsc::UnboundedSender<u64>,
    capacity_observations: Arc<CompletionCapacityObservations>,
}

pub struct State {
    api_key_state: ApiKeyState,
    credentials_provider: Arc<dyn CredentialsProvider>,
    http_client: Arc<dyn HttpClient>,
    fetched_models: Vec<ollama::Model>,
    fetch_model_task: Option<Task<Result<()>>>,
    generation: u64,
    runtime_context: Option<(Instant, ollama::RunningModelsResponse)>,
    last_runtime_capacities: HashMap<String, (Option<String>, u64)>,
    runtime_context_task: Option<Task<()>>,
    runtime_context_refresh_task: Option<Task<()>>,
    _completion_notification_task: Task<()>,
}

impl State {
    fn invalidate_runtime_context(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        self.runtime_context = None;
        self.last_runtime_capacities.clear();
        self.runtime_context_task = None;
        self.runtime_context_refresh_task = None;
    }

    fn refresh_runtime_context(&mut self, cx: &mut Context<Self>) {
        if self.runtime_context_task.is_some() {
            return;
        }
        let generation = self.generation;
        let http_client = self.http_client.clone();
        let api_url = OllamaLanguageModelProvider::api_url(cx);
        let api_key = self.api_key_state.key(&api_url);
        let headers = OllamaLanguageModelProvider::settings(cx)
            .custom_headers
            .clone();
        self.runtime_context_task = Some(cx.spawn(async move |this, cx| {
            let request =
                get_running_models(http_client.as_ref(), &api_url, api_key.as_deref(), &headers)
                    .fuse();
            let timeout = cx
                .background_executor()
                .timer(Duration::from_secs(10))
                .fuse();
            futures::pin_mut!(request, timeout);
            let result = futures::select! {
                result = request => result,
                () = timeout => Err(anyhow!("Ollama running-model query timed out")),
            };
            this.update(cx, |state, cx| {
                if state.generation != generation {
                    return;
                }
                state.runtime_context_task = None;
                match result {
                    Ok(models) => {
                        for model in state.models(cx).into_values() {
                            if let Some(capacity) =
                                models.context_length_for(&model.name, model.digest.as_deref())
                            {
                                state
                                    .last_runtime_capacities
                                    .insert(model.name, (model.digest, capacity));
                            }
                        }
                        state.runtime_context = Some((cx.background_executor().now(), models));
                    }
                    Err(error) => log::warn!("Unable to observe Ollama context capacity: {error}"),
                }
                state.runtime_context_refresh_task = Some(cx.spawn(async move |this, cx| {
                    cx.background_executor().timer(RUNTIME_CONTEXT_TTL).await;
                    this.update(cx, |state, cx| {
                        if state.generation == generation {
                            state.runtime_context_refresh_task = None;
                            state.refresh_runtime_context(cx);
                            cx.notify();
                        }
                    })
                    .ok();
                }));
                cx.notify();
            })
            .ok();
        }));
    }

    fn is_authenticated(&self) -> bool {
        !self.fetched_models.is_empty()
    }

    fn models(&self, cx: &App) -> HashMap<String, ollama::Model> {
        let settings = OllamaLanguageModelProvider::settings(cx);
        let mut models = HashMap::default();
        if settings.auto_discover {
            for model in &self.fetched_models {
                let mut model = model.clone();
                model.context_window_override = settings.context_window;
                models.insert(model.name.clone(), model);
            }
        }
        merge_settings_into_models(
            &mut models,
            &settings.available_models,
            settings.context_window,
        );
        let now = cx.background_executor().now();
        for model in models.values_mut() {
            let runtime_capacity = self
                .runtime_context
                .as_ref()
                .filter(|(observed_at, _)| {
                    now.saturating_duration_since(*observed_at) < RUNTIME_CONTEXT_TTL
                })
                .and_then(|(_, snapshot)| {
                    snapshot.context_length_for(&model.name, model.digest.as_deref())
                });
            // An expired observation is not evidence that the runner shrank to 4096.
            // Retain a labelled estimate until this model identity reports a new capacity.
            let last_capacity = self
                .last_runtime_capacities
                .get(&model.name)
                .filter(|(digest, _)| digest == &model.digest)
                .map(|(_, capacity)| *capacity);
            model.resolve_context_window(runtime_capacity.or(last_capacity));
            if runtime_capacity.is_none()
                && last_capacity.is_some()
                && model.context_window_source == ollama::ContextWindowSource::Runtime
            {
                model.context_window_source = ollama::ContextWindowSource::LastObserved;
            }
        }
        models
    }

    fn set_api_key(&mut self, api_key: Option<String>, cx: &mut Context<Self>) -> Task<Result<()>> {
        let credentials_provider = self.credentials_provider.clone();
        let api_url = OllamaLanguageModelProvider::api_url(cx);
        let task = self.api_key_state.store(
            api_url,
            api_key,
            |this| &mut this.api_key_state,
            credentials_provider,
            cx,
        );

        self.invalidate_runtime_context();
        self.fetched_models.clear();
        cx.spawn(async move |this, cx| {
            let result = task.await;
            this.update(cx, |this, cx| this.restart_fetch_models_task(cx))
                .ok();
            result
        })
    }

    fn authenticate(&mut self, cx: &mut Context<Self>) -> Task<Result<(), AuthenticateError>> {
        let credentials_provider = self.credentials_provider.clone();
        let api_url = OllamaLanguageModelProvider::api_url(cx);
        let task = self.api_key_state.load_if_needed(
            api_url,
            |this| &mut this.api_key_state,
            credentials_provider,
            cx,
        );

        // Always try to fetch models - if no API key is needed (local Ollama), it will work
        // If API key is needed and provided, it will work
        // If API key is needed and not provided, it will fail gracefully
        cx.spawn(async move |this, cx| {
            let result = task.await;
            this.update(cx, |this, cx| this.restart_fetch_models_task(cx))
                .ok();
            result
        })
    }

    fn fetch_models(&mut self, cx: &mut Context<Self>) -> Task<Result<()>> {
        let http_client = Arc::clone(&self.http_client);
        let settings = OllamaLanguageModelProvider::settings(cx);
        let api_url = OllamaLanguageModelProvider::api_url(cx);
        let api_key = self.api_key_state.key(&api_url);
        let extra_headers = settings.custom_headers.clone();
        let generation = self.generation;

        // As a proxy for the server being "authenticated", we'll check if its up by fetching the models
        cx.spawn(async move |this, cx| {
            let models = get_models(
                http_client.as_ref(),
                &api_url,
                api_key.as_deref(),
                &extra_headers,
            )
            .await?;

            let tasks = models
                .into_iter()
                // Since there is no metadata from the Ollama API
                // indicating which models are embedding models,
                // simply filter out models with "-embed" in their name
                .filter(|model| !model.name.contains("-embed"))
                .map(|model| {
                    let http_client = Arc::clone(&http_client);
                    let api_url = api_url.clone();
                    let api_key = api_key.clone();
                    let extra_headers = extra_headers.clone();
                    async move {
                        let name = model.name.as_str();
                        let model_digest = model.digest.clone();

                        show_model(
                            http_client.as_ref(),
                            &api_url,
                            api_key.as_deref(),
                            name,
                            &extra_headers,
                        )
                        .await
                        .map_or_else(
                            |error| {
                                ollama::Model::new_disabled(
                                    name,
                                    format!("Failed to fetch model from API: {error}",),
                                )
                            },
                            |model| {
                                let mut config = ollama::Model::new(
                                    name,
                                    model.context_length,
                                    Some(model.supports_tools()),
                                    Some(model.supports_vision()),
                                    Some(model.supports_thinking()),
                                );
                                config.model_default_num_ctx = model.default_num_ctx;
                                config.digest = Some(model_digest);
                                config.resolve_context_window(None);
                                config
                            },
                        )
                    }
                });

            // Rate-limit capability fetches
            // since there is an arbitrary number of models available
            let mut ollama_models: Vec<_> = futures::stream::iter(tasks)
                .buffer_unordered(5)
                .collect()
                .await;

            ollama_models.sort_by(|a, b| a.name.cmp(&b.name));

            this.update(cx, |this, cx| {
                if this.generation != generation {
                    return;
                }
                this.fetched_models = ollama_models;
                this.refresh_runtime_context(cx);
                cx.notify();
            })
        })
    }

    fn restart_fetch_models_task(&mut self, cx: &mut Context<Self>) {
        self.invalidate_runtime_context();
        let task = self.fetch_models(cx);
        self.fetch_model_task.replace(task);
    }
}

impl OllamaLanguageModelProvider {
    pub fn new(
        http_client: Arc<dyn HttpClient>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut App,
    ) -> Self {
        let (completion_notifications, mut notifications) = mpsc::unbounded();
        let this = Self {
            http_client: http_client.clone(),
            state: cx.new(|cx| {
                cx.observe_global::<SettingsStore>({
                    let mut last_settings = OllamaLanguageModelProvider::settings(cx).clone();
                    move |this: &mut State, cx| {
                        let current_settings = OllamaLanguageModelProvider::settings(cx);
                        let settings_changed = current_settings != &last_settings;
                        if settings_changed {
                            let url_changed = last_settings.api_url != current_settings.api_url;
                            last_settings = current_settings.clone();
                            this.invalidate_runtime_context();
                            if url_changed {
                                let credentials_provider = this.credentials_provider.clone();
                                let api_url = Self::api_url(cx);
                                this.api_key_state.handle_url_change(
                                    api_url,
                                    |this| &mut this.api_key_state,
                                    credentials_provider,
                                    cx,
                                );
                                this.fetched_models.clear();
                                this.authenticate(cx).detach();
                            } else {
                                this.restart_fetch_models_task(cx);
                            }
                            cx.notify();
                        }
                    }
                })
                .detach();

                let notification_task = cx.spawn(async move |this, cx| {
                    while let Some(generation) = notifications.next().await {
                        if this
                            .update(cx, |state, cx| {
                                if state.generation == generation {
                                    state.refresh_runtime_context(cx);
                                }
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                });
                State {
                    http_client,
                    fetched_models: Default::default(),
                    fetch_model_task: None,
                    api_key_state: ApiKeyState::new(Self::api_url(cx), (*API_KEY_ENV_VAR).clone()),
                    credentials_provider,
                    generation: 0,
                    runtime_context: None,
                    last_runtime_capacities: HashMap::default(),
                    runtime_context_task: None,
                    runtime_context_refresh_task: None,
                    _completion_notification_task: notification_task,
                }
            }),
            request_limiters: ModelRateLimiters::default(),
            completion_notifications,
            capacity_observations: Arc::default(),
        };
        this
    }

    /// Every model this provider offers, keyed by id: the fetched list when
    /// auto-discovery is on, with settings entries merged over it.
    fn ollama_models(&self, cx: &App) -> HashMap<String, ollama::Model> {
        self.state.read(cx).models(cx)
    }

    /// The current configuration of `model`, if this provider still offers it.
    fn config(
        &self,
        model: &LanguageModel,
        cx: &App,
    ) -> Result<ollama::Model, LanguageModelCompletionError> {
        self.ollama_models(cx)
            .remove(model.id.0.as_ref())
            .ok_or_else(|| unavailable_error(model))
    }

    fn to_ollama_request(
        config: &ollama::Model,
        request: LanguageModelRequest,
    ) -> Result<ChatRequest> {
        let num_ctx = match request.context_size {
            Some(language_model::ContextSize::Auto) => None,
            Some(context_size) => context_size.num_ctx(
                config
                    .model_context_length
                    .filter(|limit| *limit > 0)
                    .ok_or_else(|| anyhow!("Ollama model context limit is unknown"))?,
            ),
            None => config.context_window_override,
        };
        if let Some(context_window) = num_ctx {
            anyhow::ensure!(
                context_window > 0 && context_window <= i64::MAX as u64,
                "Ollama context window must be between 1 and {}",
                i64::MAX
            );
        }
        if request.contains_custom_tool_input() {
            anyhow::bail!("Ollama does not support custom tools");
        }

        let supports_vision = config.supports_vision.unwrap_or(false);

        let mut messages = Vec::with_capacity(request.messages.len());

        for mut msg in request.messages.into_iter() {
            let images = if supports_vision {
                msg.content
                    .iter()
                    .filter_map(|content| match content {
                        MessageContent::Image(image) => Some(image.source.to_string()),
                        _ => None,
                    })
                    .collect::<Vec<String>>()
            } else {
                vec![]
            };

            match msg.role {
                Role::User => {
                    for tool_result in msg
                        .content
                        .extract_if(.., |x| matches!(x, MessageContent::ToolResult(..)))
                    {
                        match tool_result {
                            MessageContent::ToolResult(tool_result) => {
                                let images = tool_result
                                    .images()
                                    .map(|image| image.source.to_string())
                                    .collect::<Vec<_>>();
                                messages.push(ChatMessage::Tool {
                                    tool_name: tool_result.tool_name.to_string(),
                                    content: tool_result.text_contents(),
                                    images: if images.is_empty() {
                                        None
                                    } else {
                                        Some(images)
                                    },
                                })
                            }
                            _ => unreachable!("Only tool result should be extracted"),
                        }
                    }
                    if !msg.content.is_empty() {
                        messages.push(ChatMessage::User {
                            content: msg.string_contents(),
                            images: if images.is_empty() {
                                None
                            } else {
                                Some(images)
                            },
                        })
                    }
                }
                Role::Assistant => {
                    let mut text_content = String::new();
                    let mut thinking = None;
                    let mut tool_calls = Vec::new();
                    for content in msg.content.into_iter() {
                        match content {
                            MessageContent::Text(text) => {
                                text_content.push_str(&text);
                            }
                            MessageContent::Thinking { text, .. } if !text.is_empty() => {
                                thinking = Some(text)
                            }
                            MessageContent::ToolUse(tool_use) => {
                                tool_calls.push(OllamaToolCall {
                                    id: tool_use.id.to_string(),
                                    function: OllamaFunctionCall {
                                        name: tool_use.name.to_string(),
                                        arguments: match tool_use.input {
                                            language_model::LanguageModelToolUseInput::Json(
                                                input,
                                            ) => input,
                                            language_model::LanguageModelToolUseInput::Text(_) => {
                                                return Err(anyhow::anyhow!(
                                                    "Ollama does not support custom tool calls"
                                                ));
                                            }
                                        },
                                    },
                                });
                            }
                            _ => (),
                        }
                    }
                    messages.push(ChatMessage::Assistant {
                        content: text_content,
                        tool_calls: Some(tool_calls),
                        images: if images.is_empty() {
                            None
                        } else {
                            Some(images)
                        },
                        thinking,
                    })
                }
                Role::System => messages.push(ChatMessage::System {
                    content: msg.string_contents(),
                }),
            }
        }
        Ok(ChatRequest {
            model: config.name.clone(),
            messages,
            keep_alive: config.keep_alive.clone(),
            stream: true,
            options: Some(ChatOptions {
                num_ctx,
                num_predict: request.max_output_tokens.map(isize::try_from).transpose()?,
                // Only send stop tokens if explicitly provided. When empty/None,
                // Ollama will use the model's default stop tokens from its Modelfile.
                // Sending an empty array would override and disable the defaults.
                stop: if request.stop.is_empty() {
                    None
                } else {
                    Some(request.stop)
                },
                temperature: request.temperature.or(Some(1.0)),
                ..Default::default()
            }),
            think: config
                .supports_thinking
                .map(|supports_thinking| supports_thinking && request.thinking_allowed),
            tools: if config.supports_tools.unwrap_or(false) {
                request
                    .tools
                    .into_iter()
                    .map(tool_into_ollama)
                    .collect::<Result<_>>()?
            } else {
                vec![]
            },
        })
    }

    fn settings(cx: &App) -> &OllamaSettings {
        &AllLanguageModelSettings::get_global(cx).ollama
    }

    fn api_url(cx: &App) -> SharedString {
        let api_url = &Self::settings(cx).api_url;
        if api_url.is_empty() {
            OLLAMA_API_URL.into()
        } else {
            SharedString::new(api_url.as_str())
        }
    }

    fn has_custom_url(cx: &App) -> bool {
        Self::settings(cx).api_url != OLLAMA_API_URL
    }
}

impl LanguageModelProviderState for OllamaLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for OllamaLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::AiOllama)
    }

    fn default_model(&self, _: &App) -> Option<LanguageModel> {
        // We shouldn't try to select default model, because it might lead to a load call for an unloaded model.
        // In a constrained environment where user might not have enough resources it'll be a bad UX to select something
        // to load by default.
        None
    }

    fn default_fast_model(&self, _: &App) -> Option<LanguageModel> {
        // See explanation for default_model.
        None
    }

    fn provided_models(&self, cx: &App) -> Vec<LanguageModel> {
        let mut models = self
            .ollama_models(cx)
            .values()
            .map(language_model)
            .collect::<Vec<_>>();
        models.sort_by_key(|model| model.name());
        models
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state.update(cx, |state, cx| state.authenticate(cx))
    }

    fn settings_view(&self, _cx: &mut App) -> Option<ProviderSettingsView> {
        let state = self.state.clone();
        Some(ProviderSettingsView::SubPage(
            SubPageProviderSettings::new(move |window, cx| {
                cx.new(|cx| ConfigurationView::new(state.clone(), window, cx))
                    .into()
            })
            .description(InlineDescription::Text(
                "Run local models on your machine with Ollama.".into(),
            )),
        ))
    }
}

impl LanguageModelClient for OllamaLanguageModelProvider {
    fn stream_completion(
        &self,
        model: &LanguageModel,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<'static, Result<LanguageModelCompletionStream, LanguageModelCompletionError>>
    {
        let config = match cx.update(|cx| self.config(model, cx)) {
            Ok(config) => config,
            Err(error) => return async move { Err(error) }.boxed(),
        };
        let request_limiter = self.request_limiters.for_model(&model.id);
        let request = match Self::to_ollama_request(&config, request) {
            Ok(request) => request,
            Err(error) => return async move { Err(error.into()) }.boxed(),
        };

        let http_client = self.http_client.clone();
        let (api_key, api_url, extra_headers, generation) =
            self.state.read_with(cx, |state, cx| {
                let api_url = OllamaLanguageModelProvider::api_url(cx);
                let extra_headers = OllamaLanguageModelProvider::settings(cx)
                    .custom_headers
                    .clone();
                (
                    state.api_key_state.key(&api_url),
                    api_url,
                    extra_headers,
                    state.generation,
                )
            });
        let notifications = self.completion_notifications.clone();
        let executor = cx.background_executor().clone();
        let model_name = config.name.clone();
        let model_digest = config.digest.clone();
        let requested_context_window = request.options.as_ref().and_then(|options| options.num_ctx);
        let capacity_observations = self.capacity_observations.clone();

        let future = request_limiter.stream(async move {
            let overlapped = capacity_observations.begin(&api_url, &model_name);
            let stream = stream_chat_completion(
                http_client.as_ref(),
                &api_url,
                api_key.as_deref(),
                request,
                &extra_headers,
            )
            .await?;
            let mut notified = false;
            let stream = stream
                .inspect(move |delta| {
                    if !notified && delta.as_ref().is_ok_and(|delta| delta.done) {
                        notified = true;
                        if let Err(error) = notifications.unbounded_send(generation) {
                            log::debug!("Ollama provider no longer observes completions: {error}");
                        }
                    }
                })
                .boxed();
            let stream = map_to_language_model_completion_events(stream).then(move |event| {
                let http_client = http_client.clone();
                let api_url = api_url.clone();
                let api_key = api_key.clone();
                let extra_headers = extra_headers.clone();
                let model_name = model_name.clone();
                let model_digest = model_digest.clone();
                let executor = executor.clone();
                let overlapped = overlapped.clone();
                async move {
                    let mut events = Vec::new();
                    if matches!(&event, Ok(LanguageModelCompletionEvent::UsageUpdate(_))) {
                        let observation = get_running_models(http_client.as_ref(), &api_url, api_key.as_deref(), &extra_headers).fuse();
                        let timeout = executor.timer(Duration::from_secs(2)).fuse();
                        futures::pin_mut!(observation, timeout);
                        let result = futures::select! {
                            result = observation => result,
                            () = timeout => Err(anyhow!("Ollama completion capacity query timed out")),
                        };
                        let capacity = match result {
                            Ok(models) => models.context_length_for(&model_name, model_digest.as_deref())
                                .filter(|capacity| !overlapped.load(Ordering::SeqCst)
                                    && requested_context_window.is_none_or(|requested| requested == *capacity)),
                            Err(error) => {
                                log::warn!("Unable to observe completion context capacity: {error}");
                                None
                            }
                        };
                        events.push(Ok(LanguageModelCompletionEvent::ContextWindow(capacity)));
                    }
                    events.push(event);
                    events
                }
            }).flat_map(stream::iter);
            Ok(stream)
        });

        future.map_ok(|f| f.boxed()).boxed()
    }
}

fn language_model(model: &ollama::Model) -> LanguageModel {
    LanguageModel {
        context_window_limit: model.model_context_length.filter(|limit| *limit > 0),
        supports_tools: model.supports_tools.unwrap_or(false),
        supports_images: model.supports_vision.unwrap_or(false),
        supports_thinking: model.supports_thinking.unwrap_or(false),
        disabled_reason: model.disabled.as_ref().map(|d| DisabledReason::new(d)),
        ..LanguageModel::new(
            LanguageModelId::from(model.name.clone()),
            LanguageModelName::from(model.display_name().to_string()),
            PROVIDER_ID,
            PROVIDER_NAME,
            format!("ollama/{}", model.id()),
            model.max_token_count(),
        )
    }
}

fn map_to_language_model_completion_events(
    stream: Pin<Box<dyn Stream<Item = anyhow::Result<ChatResponseDelta>> + Send>>,
) -> impl Stream<Item = Result<LanguageModelCompletionEvent, LanguageModelCompletionError>> {
    struct State {
        stream: Pin<Box<dyn Stream<Item = anyhow::Result<ChatResponseDelta>> + Send>>,
        used_tools: bool,
    }

    // We need to create a ToolUse and Stop event from a single
    // response from the original stream
    let stream = stream::unfold(
        State {
            stream,
            used_tools: false,
        },
        async move |mut state| {
            let response = state.stream.next().await?;

            let delta = match response {
                Ok(delta) => delta,
                Err(e) => {
                    let event = Err(LanguageModelCompletionError::from(anyhow!(e)));
                    return Some((vec![event], state));
                }
            };

            let mut events = Vec::new();

            match delta.message {
                ChatMessage::User { content, images: _ } => {
                    events.push(Ok(LanguageModelCompletionEvent::Text(content)));
                }
                ChatMessage::System { content } => {
                    events.push(Ok(LanguageModelCompletionEvent::Text(content)));
                }
                ChatMessage::Tool { content, .. } => {
                    events.push(Ok(LanguageModelCompletionEvent::Text(content)));
                }
                ChatMessage::Assistant {
                    content,
                    tool_calls,
                    images: _,
                    thinking,
                } => {
                    if let Some(text) = thinking {
                        events.push(Ok(LanguageModelCompletionEvent::Thinking {
                            text,
                            signature: None,
                        }));
                    }

                    if let Some(tool_call) = tool_calls.and_then(|v| v.into_iter().next()) {
                        let OllamaToolCall { id, function } = tool_call;
                        let event = LanguageModelCompletionEvent::ToolUse(LanguageModelToolUse {
                            id: LanguageModelToolUseId::from(id),
                            name: Arc::from(function.name),
                            raw_input: function.arguments.to_string(),
                            input: language_model::LanguageModelToolUseInput::Json(
                                function.arguments,
                            ),
                            is_input_complete: true,
                            thought_signature: None,
                        });
                        events.push(Ok(event));
                        state.used_tools = true;
                    } else if !content.is_empty() {
                        events.push(Ok(LanguageModelCompletionEvent::Text(content)));
                    }
                }
            };

            if delta.done {
                events.push(Ok(LanguageModelCompletionEvent::UsageUpdate(TokenUsage {
                    input_tokens: delta.prompt_eval_count.unwrap_or(0),
                    output_tokens: delta.eval_count.unwrap_or(0),
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                })));
                if state.used_tools {
                    state.used_tools = false;
                    events.push(Ok(LanguageModelCompletionEvent::Stop(StopReason::ToolUse)));
                } else {
                    events.push(Ok(LanguageModelCompletionEvent::Stop(StopReason::EndTurn)));
                }
            }

            Some((events, state))
        },
    );

    stream.flat_map(futures::stream::iter)
}

struct ConfigurationView {
    api_key_editor: Entity<InputField>,
    api_url_editor: Entity<InputField>,
    context_window_editor: Entity<InputField>,
    context_window_error: Option<SharedString>,
    state: Entity<State>,
}

impl ConfigurationView {
    pub fn new(state: Entity<State>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let api_key_editor = cx.new(|cx| InputField::new(window, cx, "63e02e...").label("API key"));

        let api_url_editor = cx.new(|cx| {
            let input = InputField::new(window, cx, OLLAMA_API_URL).label("API URL");
            input.set_text(&OllamaLanguageModelProvider::api_url(cx), window, cx);
            input
        });

        let context_window_editor = cx.new(|cx| {
            let input = InputField::new(window, cx, "Managed by Ollama").label("Context Window");
            if let Some(context_window) = OllamaLanguageModelProvider::settings(cx).context_window {
                input.set_text(&context_window.to_string(), window, cx);
            }
            input
        });

        cx.observe(&state, |_, _, cx| {
            cx.notify();
        })
        .detach();

        Self {
            api_key_editor,
            api_url_editor,
            context_window_editor,
            context_window_error: None,
            state,
        }
    }

    fn retry_connection(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let has_api_url = OllamaLanguageModelProvider::has_custom_url(cx);
        let has_api_key = self
            .state
            .read_with(cx, |state, _| state.api_key_state.has_key());
        if !has_api_url {
            self.save_api_url(cx);
        }
        if !has_api_key {
            self.save_api_key(&Default::default(), window, cx);
        }

        self.state.update(cx, |state, cx| {
            state.restart_fetch_models_task(cx);
        });
    }

    fn save_api_key(&mut self, _: &menu::Confirm, window: &mut Window, cx: &mut Context<Self>) {
        let api_key = self.api_key_editor.read(cx).text(cx).trim().to_string();
        if api_key.is_empty() {
            return;
        }

        // url changes can cause the editor to be displayed again
        self.api_key_editor
            .update(cx, |input, cx| input.set_text("", window, cx));

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state
                .update(cx, |state, cx| state.set_api_key(Some(api_key), cx))
                .await
        })
        .detach_and_log_err(cx);
    }

    fn reset_api_key(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.api_key_editor
            .update(cx, |input, cx| input.set_text("", window, cx));

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state
                .update(cx, |state, cx| state.set_api_key(None, cx))
                .await
        })
        .detach_and_log_err(cx);

        cx.notify();
    }

    fn save_api_url(&self, cx: &mut Context<Self>) {
        let api_url = self.api_url_editor.read(cx).text(cx).trim().to_string();
        let current_url = OllamaLanguageModelProvider::api_url(cx);
        if !api_url.is_empty() && &api_url != &current_url {
            let fs = <dyn Fs>::global(cx);
            update_settings_file(fs, cx, move |settings, _| {
                settings
                    .language_models
                    .get_or_insert_default()
                    .ollama
                    .get_or_insert_default()
                    .api_url = Some(api_url);
            });
        }
    }

    fn reset_api_url(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.api_url_editor
            .update(cx, |input, cx| input.set_text("", window, cx));
        let fs = <dyn Fs>::global(cx);
        update_settings_file(fs, cx, |settings, _cx| {
            if let Some(settings) = settings
                .language_models
                .as_mut()
                .and_then(|models| models.ollama.as_mut())
            {
                settings.api_url = Some(OLLAMA_API_URL.into());
            }
        });
        cx.notify();
    }

    fn save_context_window(&mut self, cx: &mut Context<Self>) {
        let context_window_str = self
            .context_window_editor
            .read(cx)
            .text(cx)
            .trim()
            .to_string();
        let current_context_window = OllamaLanguageModelProvider::settings(cx).context_window;

        self.context_window_error = None;
        if !context_window_str.is_empty()
            && !context_window_str
                .parse::<u64>()
                .is_ok_and(|value| value > 0 && value <= i64::MAX as u64)
        {
            self.context_window_error = Some(
                "Enter a positive context size, or leave empty to use Ollama's configuration."
                    .into(),
            );
            cx.notify();
            return;
        }
        if let Ok(context_window) = context_window_str.parse::<u64>() {
            if Some(context_window) != current_context_window {
                let fs = <dyn Fs>::global(cx);
                update_settings_file(fs, cx, move |settings, _| {
                    settings
                        .language_models
                        .get_or_insert_default()
                        .ollama
                        .get_or_insert_default()
                        .context_window = Some(context_window);
                });
            }
        } else if context_window_str.is_empty() && current_context_window.is_some() {
            let fs = <dyn Fs>::global(cx);
            update_settings_file(fs, cx, move |settings, _| {
                settings
                    .language_models
                    .get_or_insert_default()
                    .ollama
                    .get_or_insert_default()
                    .context_window = None;
            });
        }
        cx.notify();
    }

    fn reset_context_window(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.context_window_error = None;
        self.context_window_editor
            .update(cx, |input, cx| input.set_text("", window, cx));
        let fs = <dyn Fs>::global(cx);
        update_settings_file(fs, cx, |settings, _cx| {
            if let Some(settings) = settings
                .language_models
                .as_mut()
                .and_then(|models| models.ollama.as_mut())
            {
                settings.context_window = None;
            }
        });
        cx.notify();
    }

    fn render_instructions(cx: &App) -> Div {
        v_flex()
            .gap_2()
            .child(
                Label::new(
                    "Run LLMs locally on your machine with Ollama, or connect to an Ollama server. \
                Can provide access to Llama, Mistral, Gemma, and hundreds of other models.",
                )
                .color(Color::Muted),
            )
            .child(Label::new("To use local Ollama:").color(Color::Muted))
            .child(
                List::new()
                    .child(
                        ListBulletItem::new("")
                            .child(
                                Label::new("Download and install Ollama from").color(Color::Muted),
                            )
                            .child(ButtonLink::new("ollama.com", "https://ollama.com/download")),
                    )
                    .child(
                        ListBulletItem::new("")
                            .child(
                                Label::new("Start Ollama and download a model:")
                                    .color(Color::Muted),
                            )
                            .child(Label::new("ollama run gpt-oss:20b").inline_code(cx)),
                    )
                    .child(
                        ListBulletItem::new("Click 'Connect' below to start using Ollama in Zed")
                            .label_color(Color::Muted),
                    ),
            )
            .child(
                Label::new(
                    "Alternatively, you can connect to an Ollama server by specifying its \
                URL and API key (may not be required):",
                )
                .color(Color::Muted),
            )
    }

    fn render_api_key_editor(&self, cx: &Context<Self>) -> impl IntoElement {
        let state = self.state.read(cx);
        let env_var_set = state.api_key_state.is_from_env_var();
        let configured_card_label = if env_var_set {
            format!("API key set in {API_KEY_ENV_VAR_NAME} environment variable.")
        } else {
            "API key configured".to_string()
        };

        let api_key_control = if !state.api_key_state.has_key() {
            self.api_key_editor.clone().into_any_element()
        } else {
            ConfiguredApiCard::new("ollama-reset-key", configured_card_label)
                .disabled(env_var_set)
                .on_click(cx.listener(|this, _, window, cx| this.reset_api_key(window, cx)))
                .when(env_var_set, |this| {
                    this.tooltip_label(format!("To reset your API key, unset the {API_KEY_ENV_VAR_NAME} environment variable."))
                })
                .into_any_element()
        };

        v_flex()
          .on_action(cx.listener(Self::save_api_key))
          .child(api_key_control)
          .gap_1p5()
          .mb_2()
          .child(
              Label::new(
                  format!("You can also set the {API_KEY_ENV_VAR_NAME} environment variable and restart Zed.")
              )
              .size(LabelSize::Small)
              .color(Color::Muted),
          )
    }

    fn render_context_window_editor(&self, cx: &Context<Self>) -> Div {
        let settings = OllamaLanguageModelProvider::settings(cx);
        let custom_context_window_set = settings.context_window.is_some();

        if custom_context_window_set {
            h_flex()
                .p_1()
                .justify_between()
                .rounded_md()
                .border_1()
                .border_color(cx.theme().colors().border_variant)
                .bg(cx.theme().colors().background.opacity(0.5))
                .child(
                    h_flex()
                        .gap_1()
                        .child(Icon::new(IconName::Check).color(Color::Success))
                        .child(Label::new(format!(
                            "Context Window: {}",
                            settings.context_window.unwrap()
                        ))),
                )
                .child(
                    Button::new("reset-context-window", "Reset")
                        .style(ButtonStyle::Outlined)
                        .label_size(LabelSize::Small)
                        .start_icon(Icon::new(IconName::Undo).size(IconSize::Small))
                        .on_click(
                            cx.listener(|this, _, window, cx| {
                                this.reset_context_window(window, cx)
                            }),
                        ),
                )
        } else {
            v_flex()
                .on_action(
                    cx.listener(|this, _: &menu::Confirm, _window, cx| {
                        this.save_context_window(cx)
                    }),
                )
                .child(self.context_window_editor.clone())
                .gap_1p5()
                .child(
                    Label::new(
                        "Default: Managed by Ollama. Models use the server's keep-alive setting.",
                    )
                    .size(LabelSize::Small)
                    .color(Color::Muted),
                )
                .when_some(self.context_window_error.clone(), |this, error| {
                    this.child(Label::new(error).size(LabelSize::Small).color(Color::Error))
                })
        }
    }

    fn render_api_url_editor(&self, cx: &Context<Self>) -> Div {
        let api_url = OllamaLanguageModelProvider::api_url(cx);
        let custom_api_url_set = api_url != OLLAMA_API_URL;

        if custom_api_url_set {
            h_flex()
                .p_1()
                .justify_between()
                .rounded_md()
                .border_1()
                .border_color(cx.theme().colors().border_variant)
                .bg(cx.theme().colors().background.opacity(0.5))
                .child(
                    h_flex()
                        .gap_1()
                        .child(Icon::new(IconName::Check).color(Color::Success))
                        .child(Label::new(api_url)),
                )
                .child(
                    Button::new("reset-api-url", "Reset API URL")
                        .style(ButtonStyle::Outlined)
                        .label_size(LabelSize::Small)
                        .start_icon(Icon::new(IconName::Undo).size(IconSize::Small))
                        .on_click(
                            cx.listener(|this, _, window, cx| this.reset_api_url(window, cx)),
                        ),
                )
        } else {
            v_flex()
                .on_action(cx.listener(|this, _: &menu::Confirm, _window, cx| {
                    this.save_api_url(cx);
                    cx.notify();
                }))
                .gap_2()
                .child(self.api_url_editor.clone())
        }
    }
}

impl Render for ConfigurationView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let is_authenticated = self.state.read(cx).is_authenticated();

        v_flex()
            .gap_2()
            .child(Headline::new("Ollama").size(HeadlineSize::Small))
            .child(Self::render_instructions(cx))
            .child(self.render_api_url_editor(cx))
            .child(self.render_context_window_editor(cx))
            .child(v_flex().children({
                let mut models: Vec<_> = self.state.read(cx).models(cx).into_values().collect();
                models.sort_by(|left, right| left.name.cmp(&right.name));
                models.into_iter().map(|model| {
                    let source = match model.context_window_source {
                        ollama::ContextWindowSource::Explicit => "explicit setting",
                        ollama::ContextWindowSource::Runtime => "observed runtime",
                        ollama::ContextWindowSource::LastObserved => {
                            "last observed; runtime unconfirmed"
                        }
                        ollama::ContextWindowSource::ModelDefault => {
                            "model default; runtime unconfirmed"
                        }
                        ollama::ContextWindowSource::Estimate => "estimate; runtime unconfirmed",
                    };
                    let label = format!(
                        "{}: {} tokens ({source})",
                        model.display_name(),
                        model.max_tokens
                    );
                    Label::new(label).size(LabelSize::Small).color(Color::Muted)
                })
            }))
            .child(self.render_api_key_editor(cx))
            .child(Divider::horizontal())
            .child(
                h_flex()
                    .pt_2()
                    .w_full()
                    .justify_between()
                    .gap_1()
                    .child(
                        h_flex()
                            .w_full()
                            .gap_2()
                            .map(|this| {
                                if is_authenticated {
                                    this.child(
                                        Button::new("ollama-site", "Ollama")
                                            .style(ButtonStyle::OutlinedGhost)
                                            .size(ButtonSize::Medium)
                                            .end_icon(
                                                Icon::new(IconName::ArrowUpRight)
                                                    .size(IconSize::XSmall)
                                                    .color(Color::Muted),
                                            )
                                            .on_click(move |_, _, cx| cx.open_url(OLLAMA_SITE))
                                            .into_any_element(),
                                    )
                                } else {
                                    this.child(
                                        Button::new("download_ollama_button", "Download Ollama")
                                            .style(ButtonStyle::OutlinedGhost)
                                            .size(ButtonSize::Medium)
                                            .end_icon(
                                                Icon::new(IconName::ArrowUpRight)
                                                    .size(IconSize::XSmall)
                                                    .color(Color::Muted),
                                            )
                                            .on_click(move |_, _, cx| {
                                                cx.open_url(OLLAMA_DOWNLOAD_URL)
                                            })
                                            .into_any_element(),
                                    )
                                }
                            })
                            .child(
                                Button::new("view-models", "View All Models")
                                    .style(ButtonStyle::OutlinedGhost)
                                    .size(ButtonSize::Medium)
                                    .end_icon(
                                        Icon::new(IconName::ArrowUpRight)
                                            .size(IconSize::XSmall)
                                            .color(Color::Muted),
                                    )
                                    .on_click(move |_, _, cx| cx.open_url(OLLAMA_LIBRARY_URL)),
                            ),
                    )
                    .map(|this| {
                        if is_authenticated {
                            this.child(
                                ButtonLike::new("connected")
                                    .size(ButtonSize::Medium)
                                    .child(
                                        h_flex()
                                            .gap_1()
                                            .child(Icon::new(IconName::Check).color(Color::Success))
                                            .child(Label::new("Connected")),
                                    )
                                    .child(
                                        IconButton::new("refresh-models", IconName::RotateCcw)
                                            .icon_size(IconSize::Small)
                                            .tooltip(Tooltip::text("Refresh Models"))
                                            .on_click(cx.listener(|this, _, window, cx| {
                                                this.state.update(cx, |state, _| {
                                                    state.fetched_models.clear();
                                                });
                                                this.retry_connection(window, cx);
                                            })),
                                    ),
                            )
                        } else {
                            this.child(
                                Button::new("retry_ollama_models", "Connect")
                                    .style(ButtonStyle::Outlined)
                                    .size(ButtonSize::Medium)
                                    .start_icon(
                                        Icon::new(IconName::PlayOutlined).size(IconSize::XSmall),
                                    )
                                    .on_click(cx.listener(move |this, _, window, cx| {
                                        this.retry_connection(window, cx)
                                    })),
                            )
                        }
                    }),
            )
    }
}

fn merge_settings_into_models(
    models: &mut HashMap<String, ollama::Model>,
    available_models: &[AvailableModel],
    context_window: Option<u64>,
) {
    for setting_model in available_models {
        if let Some(model) = models.get_mut(&setting_model.name) {
            model.context_window_override = context_window.or(setting_model.max_tokens);
            model.resolve_context_window(None);
            model.display_name = setting_model.display_name.clone();
            model.keep_alive = setting_model.keep_alive.clone();
            model.supports_tools = setting_model.supports_tools;
            model.supports_vision = setting_model.supports_images;
            model.supports_thinking = setting_model.supports_thinking;
        } else {
            models.insert(
                setting_model.name.clone(),
                ollama::Model {
                    name: setting_model.name.clone(),
                    display_name: setting_model.display_name.clone(),
                    context_window_override: context_window.or(setting_model.max_tokens),
                    keep_alive: setting_model.keep_alive.clone(),
                    supports_tools: setting_model.supports_tools,
                    supports_vision: setting_model.supports_images,
                    supports_thinking: setting_model.supports_thinking,
                    ..ollama::Model::new(&setting_model.name, None, None, None, None)
                },
            );
        }
    }
    for model in models.values_mut() {
        model.resolve_context_window(None);
    }
}

fn tool_into_ollama(tool: LanguageModelRequestTool) -> Result<ollama::OllamaTool> {
    let input_schema = match tool.input {
        language_model::LanguageModelRequestToolInput::Function { input_schema, .. } => {
            input_schema
        }
        language_model::LanguageModelRequestToolInput::Custom { .. } => {
            anyhow::bail!("Ollama does not support custom tools");
        }
    };
    Ok(ollama::OllamaTool::Function {
        function: OllamaFunctionTool {
            name: tool.name,
            description: Some(tool.description),
            parameters: Some(input_schema),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_context_size_uses_model_limit_and_auto_omits_num_ctx() -> Result<()> {
        use language_model::ContextSize;

        let mut model = ollama::Model::new("model", Some(131072), None, None, None);
        model.context_window_override = Some(8192);
        model.resolve_context_window(Some(32768));

        assert_eq!(language_model(&model).context_window_limit, Some(131072));

        for (context_size, expected) in [
            (ContextSize::Auto, None),
            (ContextSize::Quarter, Some(32768)),
            (ContextSize::Half, Some(65536)),
            (ContextSize::ThreeQuarters, Some(98304)),
            (ContextSize::Full, Some(131072)),
        ] {
            let request = LanguageModelRequest {
                context_size: Some(context_size),
                ..Default::default()
            };
            let json = serde_json::to_value(OllamaLanguageModelProvider::to_ollama_request(
                &model, request,
            )?)?;
            assert_eq!(
                json["options"]
                    .get("num_ctx")
                    .and_then(|value| value.as_u64()),
                expected
            );
            assert_eq!(json["options"].get("num_ctx").is_none(), expected.is_none());
        }
        Ok(())
    }

    #[test]
    fn requests_only_override_explicit_context_and_residency() -> Result<()> {
        let mut model = ollama::Model::new("model", Some(131072), None, None, None);
        model.model_default_num_ctx = Some(8192);
        model.resolve_context_window(Some(32768));
        let request = LanguageModelRequest {
            max_output_tokens: Some(128),
            ..Default::default()
        };
        let json = serde_json::to_value(OllamaLanguageModelProvider::to_ollama_request(
            &model,
            request.clone(),
        )?)?;
        assert!(json.get("keep_alive").is_none());
        assert!(json["options"].get("num_ctx").is_none());
        assert_eq!(json["options"]["num_predict"], 128);
        model.context_window_override = Some(8192);
        model.keep_alive = Some(ollama::KeepAlive::Seconds(0));
        let json = serde_json::to_value(OllamaLanguageModelProvider::to_ollama_request(
            &model,
            request.clone(),
        )?)?;
        assert_eq!(json["keep_alive"], 0);
        assert_eq!(json["options"]["num_ctx"], 8192);
        for invalid in [0, u64::MAX] {
            model.context_window_override = Some(invalid);
            assert!(
                OllamaLanguageModelProvider::to_ollama_request(&model, request.clone()).is_err()
            );
        }
        Ok(())
    }

    #[test]
    fn explicit_global_context_wins_and_metadata_only_entries_do_not_override() -> Result<()> {
        let settings: Vec<AvailableModel> = serde_json::from_value(serde_json::json!([
            {"name":"model","max_tokens":16384},
            {"name":"manual-model","display_name":"Manual"}
        ]))?;
        let mut models = HashMap::default();
        models.insert(
            "model".into(),
            ollama::Model::new("model", Some(131072), Some(true), None, None),
        );
        merge_settings_into_models(&mut models, &settings, Some(8192));
        assert_eq!(
            models
                .get("model")
                .and_then(|model| model.context_window_override),
            Some(8192)
        );
        assert_eq!(
            models.get("manual-model").map(|model| model.max_tokens),
            Some(8192)
        );
        merge_settings_into_models(&mut models, &settings, None);
        assert_eq!(
            models
                .get("model")
                .and_then(|model| model.context_window_override),
            Some(16384)
        );
        assert_eq!(
            models
                .get("manual-model")
                .and_then(|model| model.context_window_override),
            None
        );
        assert_eq!(
            models.get("manual-model").map(|model| model.max_tokens),
            Some(4096)
        );
        Ok(())
    }

    struct TestCredentialsProvider;

    impl CredentialsProvider for TestCredentialsProvider {
        fn read_credentials<'a>(
            &'a self,
            _: &'a str,
            _: &'a AsyncApp,
        ) -> Pin<Box<dyn std::future::Future<Output = Result<Option<(String, Vec<u8>)>>> + 'a>>
        {
            Box::pin(async { Ok(None) })
        }
        fn write_credentials<'a>(
            &'a self,
            _: &'a str,
            _: &'a str,
            _: &'a [u8],
            _: &'a AsyncApp,
        ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + 'a>> {
            Box::pin(async { Ok(()) })
        }
        fn delete_credentials<'a>(
            &'a self,
            _: &'a str,
            _: &'a AsyncApp,
        ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + 'a>> {
            Box::pin(async { Ok(()) })
        }
    }

    #[gpui::test]
    fn idle_runtime_refresh_preserves_capacity_and_invalidation_restores_estimate(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let http_client = http_client::FakeHttpClient::create(|request| async move {
            assert_eq!(request.method(), http_client::Method::GET);
            assert_eq!(request.uri().path(), "/api/ps");
            Ok(http_client::Response::builder()
                .status(200)
                .body(http_client::AsyncBody::from(
                    serde_json::json!({"models":[{"name":"model:latest","context_length":32768}]})
                        .to_string(),
                ))?)
        });
        let provider = cx.update(|cx| {
            OllamaLanguageModelProvider::new(http_client, Arc::new(TestCredentialsProvider), cx)
        });
        provider.state.update(cx, |state, cx| {
            state.fetched_models = vec![ollama::Model::new(
                "model:latest",
                Some(131072),
                None,
                None,
                None,
            )];
            state.refresh_runtime_context(cx);
        });
        cx.run_until_parked();
        cx.update(|cx| {
            assert_eq!(
                provider
                    .provided_models(cx)
                    .first()
                    .map(|model| model.max_token_count),
                Some(32768)
            )
        });
        cx.executor().advance_clock(RUNTIME_CONTEXT_TTL);
        cx.run_until_parked();
        cx.update(|cx| {
            assert_eq!(
                provider
                    .provided_models(cx)
                    .first()
                    .map(|model| model.max_token_count),
                Some(32768)
            )
        });
        for _ in 0..5 {
            cx.executor().advance_clock(RUNTIME_CONTEXT_TTL);
            cx.run_until_parked();
            cx.update(|cx| {
                let config = provider.ollama_models(cx);
                let model = config.get("model:latest").expect("discovered model");
                assert_eq!(model.max_tokens, 32768);
                assert_eq!(
                    model.context_window_source,
                    ollama::ContextWindowSource::Runtime
                );
            });
        }
        provider
            .state
            .update(cx, |state, cx| state.refresh_runtime_context(cx));
        cx.run_until_parked();
        provider
            .state
            .update(cx, |state, _| state.invalidate_runtime_context());
        cx.update(|cx| {
            assert_eq!(
                provider
                    .provided_models(cx)
                    .first()
                    .map(|model| model.max_token_count),
                Some(4096)
            )
        });
        drop(provider);
        cx.run_until_parked();
    }

    #[gpui::test]
    fn runtime_queries_coalesce_timeout_and_ignore_old_completions(cx: &mut gpui::TestAppContext) {
        use std::sync::atomic::{AtomicUsize, Ordering};

        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let requests = Arc::new(AtomicUsize::new(0));
        let http_client = http_client::FakeHttpClient::create({
            let requests = requests.clone();
            move |_| {
                requests.fetch_add(1, Ordering::SeqCst);
                futures::future::pending()
            }
        });
        let provider = cx.update(|cx| {
            OllamaLanguageModelProvider::new(http_client, Arc::new(TestCredentialsProvider), cx)
        });
        provider.state.update(cx, |state, cx| {
            state.refresh_runtime_context(cx);
            state.refresh_runtime_context(cx);
        });
        cx.run_until_parked();
        assert_eq!(requests.load(Ordering::SeqCst), 1);
        cx.executor().advance_clock(Duration::from_secs(10));
        cx.run_until_parked();
        provider.state.read_with(cx, |state, _| {
            assert!(state.runtime_context_task.is_none());
            assert!(state.runtime_context.is_none());
        });
        provider
            .state
            .update(cx, |state, cx| state.refresh_runtime_context(cx));
        cx.run_until_parked();
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        provider
            .state
            .update(cx, |state, _| state.invalidate_runtime_context());
        provider
            .completion_notifications
            .unbounded_send(0)
            .expect("provider is alive");
        cx.run_until_parked();
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        provider
            .state
            .read_with(cx, |state, _| assert!(state.runtime_context_task.is_none()));
        drop(provider);
        cx.run_until_parked();
    }

    #[gpui::test]
    fn failed_runtime_query_does_not_extend_observation(cx: &mut gpui::TestAppContext) {
        use std::sync::atomic::{AtomicUsize, Ordering};

        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let requests = Arc::new(AtomicUsize::new(0));
        let http_client = http_client::FakeHttpClient::create(move |_| {
            let first = requests.fetch_add(1, Ordering::SeqCst) == 0;
            async move {
                Ok(http_client::Response::builder()
                    .status(if first { 200 } else { 404 })
                    .body(http_client::AsyncBody::from("{\"models\":[]}"))?)
            }
        });
        let provider = cx.update(|cx| {
            OllamaLanguageModelProvider::new(http_client, Arc::new(TestCredentialsProvider), cx)
        });
        provider
            .state
            .update(cx, |state, cx| state.refresh_runtime_context(cx));
        cx.run_until_parked();
        cx.executor().advance_clock(Duration::from_secs(30));
        provider
            .state
            .update(cx, |state, cx| state.refresh_runtime_context(cx));
        cx.run_until_parked();
        provider.state.read_with(cx, |state, _| {
            let observed_at = state.runtime_context.as_ref().expect("snapshot").0;
            assert_eq!(
                cx.executor().now().saturating_duration_since(observed_at),
                Duration::from_secs(30)
            );
        });
        cx.executor().advance_clock(Duration::from_secs(30));
        cx.run_until_parked();
        provider.state.read_with(cx, |state, _| {
            let observed_at = state.runtime_context.as_ref().expect("snapshot").0;
            assert_eq!(
                cx.executor().now().saturating_duration_since(observed_at),
                RUNTIME_CONTEXT_TTL
            );
        });
        drop(provider);
        cx.run_until_parked();
    }

    #[gpui::test]
    fn idle_unload_or_disconnect_retains_budget_but_new_capacity_and_identity_win(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::sync::atomic::{AtomicUsize, Ordering};

        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let response_mode = Arc::new(AtomicUsize::new(0));
        let http_client = http_client::FakeHttpClient::create({
            let response_mode = response_mode.clone();
            move |_| {
                let mode = response_mode.load(Ordering::SeqCst);
                async move {
                    let body = match mode {
                        0 => {
                            serde_json::json!({"models":[{"name":"model:latest", "digest":"original", "context_length":32768}]})
                        }
                        3 => {
                            serde_json::json!({"models":[{"name":"model:latest", "digest":"original", "context_length":8192}]})
                        }
                        _ => serde_json::json!({"models":[]}),
                    };
                    Ok(http_client::Response::builder()
                        .status(if mode == 2 { 503 } else { 200 })
                        .body(http_client::AsyncBody::from(body.to_string()))?)
                }
            }
        });
        let provider = cx.update(|cx| {
            OllamaLanguageModelProvider::new(http_client, Arc::new(TestCredentialsProvider), cx)
        });
        provider.state.update(cx, |state, cx| {
            let mut model = ollama::Model::new("model:latest", Some(131072), None, None, None);
            model.digest = Some("original".into());
            state.fetched_models = vec![model];
            state.refresh_runtime_context(cx);
        });
        cx.run_until_parked();
        for mode in [1, 2, 2, 1] {
            response_mode.store(mode, Ordering::SeqCst);
            cx.executor().advance_clock(RUNTIME_CONTEXT_TTL);
            cx.run_until_parked();
            cx.update(|cx| {
                let models = provider.ollama_models(cx);
                let model = models.get("model:latest").expect("model");
                assert_eq!(model.max_tokens, 32768);
                assert_eq!(
                    model.context_window_source,
                    ollama::ContextWindowSource::LastObserved
                );
                assert!(model.context_window_override.is_none());
            });
        }
        response_mode.store(3, Ordering::SeqCst);
        cx.executor().advance_clock(RUNTIME_CONTEXT_TTL);
        cx.run_until_parked();
        cx.update(|cx| assert_eq!(provider.provided_models(cx).remove(0).max_token_count, 8192));
        provider.state.update(cx, |state, _| {
            state.fetched_models.first_mut().expect("model").digest = Some("replacement".into());
        });
        cx.update(|cx| assert_eq!(provider.provided_models(cx).remove(0).max_token_count, 4096));
        drop(provider);
        cx.run_until_parked();
    }

    #[gpui::test]
    async fn completed_chat_observes_capacity_before_usage(cx: &mut gpui::TestAppContext) {
        use futures::{AsyncReadExt, TryStreamExt};
        use std::sync::atomic::{AtomicUsize, Ordering};

        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let queries = Arc::new(AtomicUsize::new(0));
        let http_client = http_client::FakeHttpClient::create({
            let queries = queries.clone();
            move |mut request| {
                let queries = queries.clone();
                async move {
                    let body = match request.uri().path() {
                        "/api/chat" => {
                            let mut body = String::new();
                            request.body_mut().read_to_string(&mut body).await?;
                            let json: serde_json::Value = serde_json::from_str(&body)?;
                            assert!(json.get("keep_alive").is_none());
                            assert!(json["options"].get("num_ctx").is_none());
                            [
                                serde_json::json!({"model":"model", "created_at":"", "done":false,
                                    "message":{"role":"assistant","content":"hello"}}),
                                serde_json::json!({"model":"model", "created_at":"", "done":true,
                                    "prompt_eval_count":12,"eval_count":3,
                                    "message":{"role":"assistant","content":"", "thinking":"reasoning",
                                        "tool_calls":[{"id":"call", "function":{"name":"example", "arguments":{}}}]}}),
                            ].into_iter().map(|delta| delta.to_string()).collect::<Vec<_>>().join("\n")
                        }
                        "/api/ps" => {
                            queries.fetch_add(1, Ordering::SeqCst);
                            serde_json::json!({"models":[{"name":"model:latest","context_length":8192}]}).to_string()
                        }
                        path => panic!("Unexpected endpoint {path}"),
                    };
                    Ok(http_client::Response::builder()
                        .status(200)
                        .body(http_client::AsyncBody::from(body))?)
                }
            }
        });
        let provider = cx.update(|cx| {
            OllamaLanguageModelProvider::new(http_client, Arc::new(TestCredentialsProvider), cx)
        });
        provider.state.update(cx, |state, _| {
            state.fetched_models = vec![ollama::Model::new(
                "model:latest",
                Some(131072),
                Some(true),
                None,
                None,
            )];
        });
        let model = cx.update(|cx| provider.provided_models(cx).remove(0));
        let events: Vec<_> = provider
            .stream_completion(&model, LanguageModelRequest::default(), &cx.to_async())
            .await
            .expect("chat stream")
            .try_collect()
            .await
            .expect("completion events");
        assert_eq!(events.len(), 6);
        assert!(matches!(&events[0], LanguageModelCompletionEvent::Text(text) if text == "hello"));
        assert!(
            matches!(&events[1], LanguageModelCompletionEvent::Thinking { text, .. } if text == "reasoning")
        );
        assert!(
            matches!(&events[2], LanguageModelCompletionEvent::ToolUse(tool) if tool.name.as_ref() == "example")
        );
        assert!(matches!(
            &events[3],
            LanguageModelCompletionEvent::ContextWindow(Some(8192))
        ));
        assert!(
            matches!(&events[4], LanguageModelCompletionEvent::UsageUpdate(usage) if usage.input_tokens == 12 && usage.output_tokens == 3)
        );
        assert!(matches!(
            &events[5],
            LanguageModelCompletionEvent::Stop(StopReason::ToolUse)
        ));
        cx.run_until_parked();
        assert_eq!(queries.load(Ordering::SeqCst), 2);
        cx.update(|cx| assert_eq!(provider.provided_models(cx).remove(0).max_token_count, 8192));
        drop(provider);
        cx.run_until_parked();
    }

    #[gpui::test]
    async fn completed_chat_keeps_usage_when_capacity_is_unknown(cx: &mut gpui::TestAppContext) {
        use futures::TryStreamExt;
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let http_client = http_client::FakeHttpClient::create(|request| async move {
            match request.uri().path() {
                "/api/chat" => Ok(http_client::Response::builder().status(200).body(
                    http_client::AsyncBody::from(
                        serde_json::json!({
                            "model":"model", "created_at":"", "done":true,
                            "prompt_eval_count":12,"eval_count":3,
                            "message":{"role":"assistant","content":"answer"}
                        })
                        .to_string(),
                    ),
                )?),
                "/api/ps" => Err(anyhow!("runtime observation unavailable")),
                path => panic!("Unexpected endpoint {path}"),
            }
        });
        let provider = cx.update(|cx| {
            OllamaLanguageModelProvider::new(http_client, Arc::new(TestCredentialsProvider), cx)
        });
        provider.state.update(cx, |state, _| {
            state.fetched_models = vec![ollama::Model::new(
                "model:latest",
                Some(131072),
                Some(true),
                None,
                None,
            )];
        });
        let model = cx.update(|cx| provider.provided_models(cx).remove(0));
        let events: Vec<_> = provider
            .stream_completion(&model, LanguageModelRequest::default(), &cx.to_async())
            .await
            .expect("stream")
            .try_collect()
            .await
            .expect("events");
        assert!(matches!(
            events.get(1),
            Some(LanguageModelCompletionEvent::ContextWindow(None))
        ));
        assert!(
            matches!(events.get(2), Some(LanguageModelCompletionEvent::UsageUpdate(usage)) if usage.input_tokens == 12)
        );
        drop(provider);
        cx.run_until_parked();
    }

    #[test]
    fn capacity_observations_remember_overlap_and_release_dropped_streams() {
        let observations = CompletionCapacityObservations::default();
        let first = observations.begin("server", "model");
        let unrelated = observations.begin("server", "other");
        let other_server = observations.begin("other-server", "model");
        assert!(!first.load(Ordering::SeqCst));
        let second = observations.begin("server", "model:latest");
        assert!(first.load(Ordering::SeqCst));
        assert!(second.load(Ordering::SeqCst));
        assert!(!unrelated.load(Ordering::SeqCst));
        assert!(!other_server.load(Ordering::SeqCst));
        drop(second);
        assert!(first.load(Ordering::SeqCst));
        drop(first);
        assert!(!observations.begin("server", "model").load(Ordering::SeqCst));
    }

    #[gpui::test]
    async fn overlapping_completions_do_not_attribute_runner_capacity(
        cx: &mut gpui::TestAppContext,
    ) {
        use futures::{AsyncReadExt, TryStreamExt};
        use std::sync::atomic::AtomicU64;

        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        let runner_capacity = Arc::new(AtomicU64::new(32768));
        let http_client = http_client::FakeHttpClient::create({
            let runner_capacity = runner_capacity.clone();
            move |mut request| {
                let runner_capacity = runner_capacity.clone();
                async move {
                    let body = match request.uri().path() {
                        "/api/chat" => {
                            let mut body = String::new();
                            request.body_mut().read_to_string(&mut body).await?;
                            let body: serde_json::Value = serde_json::from_str(&body)?;
                            let capacity = body["options"]["num_ctx"].as_u64().unwrap_or(32768);
                            runner_capacity.store(capacity, Ordering::SeqCst);
                            [
                                serde_json::json!({"model":"model", "created_at":"", "done":false,
                                    "message":{"role":"assistant","content":format!("generated at {capacity}")}}),
                                serde_json::json!({"model":"model", "created_at":"", "done":true,
                                    "prompt_eval_count":12000,"eval_count":100,
                                    "message":{"role":"assistant","content":""}}),
                            ].into_iter().map(|delta| delta.to_string()).collect::<Vec<_>>().join("\n")
                        }
                        "/api/ps" => serde_json::json!({"models":[{"name":"model:latest",
                            "context_length":runner_capacity.load(Ordering::SeqCst)}]})
                        .to_string(),
                        path => panic!("Unexpected endpoint {path}"),
                    };
                    Ok(http_client::Response::builder()
                        .status(200)
                        .body(http_client::AsyncBody::from(body))?)
                }
            }
        });
        let provider = cx.update(|cx| {
            OllamaLanguageModelProvider::new(http_client, Arc::new(TestCredentialsProvider), cx)
        });
        provider.state.update(cx, |state, _| {
            state.fetched_models = vec![ollama::Model::new(
                "model:latest",
                Some(131072),
                Some(true),
                None,
                None,
            )];
        });
        let model = cx.update(|cx| provider.provided_models(cx).remove(0));
        for context_size in [
            language_model::ContextSize::Auto,
            language_model::ContextSize::Quarter,
        ] {
            let mut first = provider
                .stream_completion(
                    &model,
                    LanguageModelRequest {
                        context_size: Some(context_size),
                        ..Default::default()
                    },
                    &cx.to_async(),
                )
                .await
                .expect("first stream");
            assert!(matches!(first.next().await.expect("text").expect("event"),
                LanguageModelCompletionEvent::Text(text) if text == "generated at 32768"));
            let second: Vec<_> = provider
                .stream_completion(
                    &model,
                    LanguageModelRequest {
                        context_size: Some(language_model::ContextSize::Half),
                        ..Default::default()
                    },
                    &cx.to_async(),
                )
                .await
                .expect("second stream")
                .try_collect()
                .await
                .expect("second events");
            let first: Vec<_> = first.try_collect().await.expect("first events");
            for events in [&first, &second] {
                assert!(events.iter().any(|event| matches!(
                    event,
                    LanguageModelCompletionEvent::ContextWindow(None)
                )));
                assert!(events.iter().any(|event| matches!(event, LanguageModelCompletionEvent::UsageUpdate(usage) if usage.input_tokens == 12000)));
                assert!(!events.iter().any(|event| matches!(
                    event,
                    LanguageModelCompletionEvent::ContextWindow(Some(_))
                )));
            }
        }
        drop(provider);
        cx.run_until_parked();
    }

    #[test]
    fn test_merge_settings_preserves_display_names_for_similar_models() {
        // Regression test for https://github.com/zed-industries/zed/issues/43646
        // When multiple models share the same base name (e.g., qwen2.5-coder:1.5b and qwen2.5-coder:3b),
        // each model should get its own display_name from settings, not a random one.

        let mut models: HashMap<String, ollama::Model> = HashMap::default();
        models.insert(
            "qwen2.5-coder:1.5b".to_string(),
            ollama::Model {
                name: "qwen2.5-coder:1.5b".to_string(),
                display_name: None,
                max_tokens: 4096,
                keep_alive: None,
                supports_tools: None,
                supports_vision: None,
                supports_thinking: None,
                disabled: None,
                ..Default::default()
            },
        );
        models.insert(
            "qwen2.5-coder:3b".to_string(),
            ollama::Model {
                name: "qwen2.5-coder:3b".to_string(),
                display_name: None,
                max_tokens: 4096,
                keep_alive: None,
                supports_tools: None,
                supports_vision: None,
                supports_thinking: None,
                disabled: None,
                ..Default::default()
            },
        );

        let available_models = vec![
            AvailableModel {
                name: "qwen2.5-coder:1.5b".to_string(),
                display_name: Some("QWEN2.5 Coder 1.5B".to_string()),
                max_tokens: Some(5000),
                keep_alive: None,
                supports_tools: Some(true),
                supports_images: None,
                supports_thinking: None,
            },
            AvailableModel {
                name: "qwen2.5-coder:3b".to_string(),
                display_name: Some("QWEN2.5 Coder 3B".to_string()),
                max_tokens: Some(6000),
                keep_alive: None,
                supports_tools: Some(true),
                supports_images: None,
                supports_thinking: None,
            },
        ];

        merge_settings_into_models(&mut models, &available_models, None);

        let model_1_5b = models
            .get("qwen2.5-coder:1.5b")
            .expect("1.5b model missing");
        let model_3b = models.get("qwen2.5-coder:3b").expect("3b model missing");

        assert_eq!(
            model_1_5b.display_name,
            Some("QWEN2.5 Coder 1.5B".to_string()),
            "1.5b model should have its own display_name"
        );
        assert_eq!(model_1_5b.max_tokens, 5000);

        assert_eq!(
            model_3b.display_name,
            Some("QWEN2.5 Coder 3B".to_string()),
            "3b model should have its own display_name"
        );
        assert_eq!(model_3b.max_tokens, 6000);
    }
}
