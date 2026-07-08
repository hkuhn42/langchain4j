package dev.langchain4j.model.openaiofficial;

import static dev.langchain4j.internal.Utils.copy;
import static dev.langchain4j.internal.Utils.getOrDefault;
import static dev.langchain4j.model.openaiofficial.InternalOpenAiOfficialHelper.aiMessageFrom;
import static dev.langchain4j.model.openaiofficial.InternalOpenAiOfficialHelper.finishReasonFrom;
import static dev.langchain4j.model.openaiofficial.InternalOpenAiOfficialHelper.fromOpenAiResponseFormat;
import static dev.langchain4j.model.openaiofficial.InternalOpenAiOfficialHelper.toOpenAiChatCompletionCreateParams;
import static dev.langchain4j.model.openaiofficial.InternalOpenAiOfficialHelper.tokenUsageFrom;
import static dev.langchain4j.model.openaiofficial.InternalOpenAiOfficialHelper.validate;
import static dev.langchain4j.model.openaiofficial.setup.OpenAiOfficialSetup.detectModelProvider;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.Proxy;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.stream.Collectors;

import com.fasterxml.jackson.databind.JsonNode;
import com.openai.azure.AzureOpenAIServiceVersion;
import com.openai.client.OpenAIClient;
import com.openai.core.ObjectMappers;
import com.openai.core.http.HttpResponse;
import com.openai.credential.Credential;
import com.openai.models.batches.Batch;
import com.openai.models.batches.BatchCreateParams;
import com.openai.models.batches.BatchListParams;
import com.openai.models.chat.completions.ChatCompletion;
import com.openai.models.chat.completions.ChatCompletionCreateParams;
import com.openai.models.files.FileCreateParams;
import com.openai.models.files.FileObject;
import com.openai.models.files.FilePurpose;

import dev.langchain4j.Experimental;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.ModelProvider;
import dev.langchain4j.model.TokenCountEstimator;
import dev.langchain4j.model.batch.BatchError;
import dev.langchain4j.model.batch.BatchItemResult;
import dev.langchain4j.model.batch.BatchPage;
import dev.langchain4j.model.batch.BatchPagination;
import dev.langchain4j.model.batch.BatchRequest;
import dev.langchain4j.model.batch.BatchResponse;
import dev.langchain4j.model.batch.BatchState;
import dev.langchain4j.model.chat.BatchChatModel;
import dev.langchain4j.model.chat.Capability;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.request.DefaultChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;

/**
 * Batch chat model implementation using the official OpenAI Java SDK and the Chat Completions Batch API.
 */
@Experimental
public class OpenAiOfficialBatchChatModel extends OpenAiOfficialBaseChatModel implements BatchChatModel {
    
    private static final String CHAT_COMPLETIONS_BATCH_ENDPOINT = "/v1/chat/completions";
    private static final String CUSTOM_ID_PREFIX                = "request-";
    
    public OpenAiOfficialBatchChatModel(Builder builder) {
        if (builder.openAIClient == null) {
            init(
                builder.baseUrl,
                builder.apiKey,
                builder.credential,
                builder.microsoftFoundryDeploymentName,
                builder.azureOpenAIServiceVersion,
                builder.organizationId,
                builder.isMicrosoftFoundry,
                builder.isGitHubModels,
                builder.defaultRequestParameters,
                builder.modelName,
                builder.temperature,
                builder.topP,
                builder.stop,
                builder.maxCompletionTokens,
                builder.presencePenalty,
                builder.frequencyPenalty,
                builder.logitBias,
                builder.responseFormat,
                builder.strictJsonSchema,
                builder.seed,
                builder.user,
                builder.strictTools,
                builder.parallelToolCalls,
                builder.store,
                builder.metadata,
                builder.serviceTier,
                builder.timeout,
                builder.maxRetries,
                builder.proxy,
                builder.tokenCountEstimator,
                builder.customHeaders,
                builder.listeners,
                builder.capabilities,
                false);
        }
        else {
            this.client = builder.openAIClient;
            initWithoutCreatingClient(builder);
        }
        this.modelName = builder.modelName;
    }
    
    @Override
    public BatchResponse<ChatResponse> submit(BatchRequest<ChatRequest> request) {
        Path jsonlFile = null;
        try {
            jsonlFile = Files.createTempFile("langchain4j-openai-batch-chat-", ".jsonl");
            writeBatchFile(jsonlFile, request.requests());
            
            FileObject file = client.files()
                .create(FileCreateParams.builder()
                    .file(jsonlFile)
                    .purpose(FilePurpose.BATCH)
                    .build());
            
            Batch batch = client.batches()
                .create(BatchCreateParams.builder()
                    .inputFileId(file.id())
                    .endpoint(BatchCreateParams.Endpoint.V1_CHAT_COMPLETIONS)
                    .completionWindow(BatchCreateParams.CompletionWindow._24H)
                    .build());
            
            return toBatchResponse(batch, false);
        }
        catch (IOException e) {
            throw new RuntimeException("Failed to create OpenAI batch request file", e);
        }
        finally {
            if (jsonlFile != null) {
                try {
                    Files.deleteIfExists(jsonlFile);
                }
                catch (IOException ignored) {
                    // Best effort cleanup of a temporary request file.
                }
            }
        }
    }
    
    @Override
    public BatchResponse<ChatResponse> retrieve(String batchId) {
        return toBatchResponse(client.batches().retrieve(batchId), true);
    }
    
    @Override
    public void cancel(String batchId) {
        client.batches().cancel(batchId);
    }
    
    @Override
    public BatchPage<ChatResponse> list(BatchPagination pagination) {
        BatchListParams.Builder paramsBuilder = BatchListParams.builder();
        if (pagination != null) {
            if (pagination.pageSize() != null) {
                paramsBuilder.limit(pagination.pageSize());
            }
            if (pagination.pageToken() != null) {
                paramsBuilder.after(pagination.pageToken());
            }
        }
        
        var page = client.batches().list(paramsBuilder.build());
        List<BatchResponse<ChatResponse>> batches = page.data().stream().map(batch -> toBatchResponse(batch, false)).collect(Collectors.toList());
        
        String nextPageToken = null;
        if (page.hasMore().orElse(false) && !page.data().isEmpty()) {
            nextPageToken = page.data().get(page.data().size() - 1).id();
        }
        return new BatchPage<>(batches, nextPageToken);
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    private void writeBatchFile(Path path, List<ChatRequest> requests) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            for (int i = 0; i < requests.size(); i++) {
                ChatRequest request = prepareRequest(requests.get(i));
                OpenAiOfficialChatRequestParameters parameters = (OpenAiOfficialChatRequestParameters) request.parameters();
                ChatCompletionCreateParams params = toOpenAiChatCompletionCreateParams(request, parameters, strictTools, strictJsonSchema)
                    .build();
                
                JsonNode body = ObjectMappers.jsonMapper().valueToTree(params._body());
                writer.write(ObjectMappers.jsonMapper()
                    .writeValueAsString(batchInputLine(CUSTOM_ID_PREFIX + i, body)));
                writer.newLine();
            }
        }
    }
    
    private ChatRequest prepareRequest(ChatRequest request) {
        ChatRequest prepared = ChatRequest.builder()
            .messages(request.messages())
            .parameters(defaultRequestParameters.overrideWith(request.parameters()))
            .build();
        
        OpenAiOfficialChatRequestParameters parameters = (OpenAiOfficialChatRequestParameters) prepared.parameters();
        validate(parameters);
        
        if ((modelProvider.equals(ModelProvider.MICROSOFT_FOUNDRY)
            || modelProvider.equals(ModelProvider.GITHUB_MODELS))
            && !parameters.modelName().equals(this.modelName)) {
            throw new UnsupportedFeatureException("Modifying the modelName is not supported");
        }
        
        return prepared;
    }
    
    private BatchResponse<ChatResponse> toBatchResponse(Batch batch, boolean includeResults) {
        BatchState state = toBatchState(batch.status());
        BatchResponse.Builder<ChatResponse> builder = BatchResponse.<ChatResponse> builder().batchId(batch.id()).state(state);
        
        if (includeResults && state.isTerminal()) {
            builder.results(readResults(batch));
        }
        else if (state == BatchState.FAILED && batch.errors().isPresent()) {
            builder.results(batch.errors().get().data().orElse(List.of()).stream()
                .map(OpenAiOfficialBatchChatModel::toBatchError)
                .map(BatchItemResult::<ChatResponse> failure)
                .collect(Collectors.toList()));
        }
        
        return builder.build();
    }
    
    private List<BatchItemResult<ChatResponse>> readResults(Batch batch) {
        TreeMap<Integer, BatchItemResult<ChatResponse>> results = new TreeMap<>();
        batch.outputFileId().ifPresent(fileId -> readResultFile(fileId, results));
        batch.errorFileId().ifPresent(fileId -> readResultFile(fileId, results));
        if (!results.isEmpty()) {
            return new ArrayList<>(results.values());
        }
        return batch.errors()
            .flatMap(Batch.Errors::data)
            .map(errors -> errors.stream()
                .map(OpenAiOfficialBatchChatModel::toBatchError)
                .map(BatchItemResult::<ChatResponse> failure)
                .collect(Collectors.toList()))
            .orElse(List.of());
    }
    
    private void readResultFile(String fileId, TreeMap<Integer, BatchItemResult<ChatResponse>> results) {
        try (HttpResponse response = client.files().content(fileId);
            BufferedReader reader = new BufferedReader(new InputStreamReader(response.body(), StandardCharsets.UTF_8))) {
            
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.isBlank()) {
                    BatchOutputLine outputLine = parseOutputLine(line);
                    results.put(outputLine.index(), outputLine.result());
                }
            }
        }
        catch (IOException e) {
            throw new RuntimeException("Failed to read OpenAI batch result file " + fileId, e);
        }
    }
    
    static BatchOutputLine parseOutputLine(String line) throws IOException {
        JsonNode root = ObjectMappers.jsonMapper().readTree(line);
        int index = customIdIndex(root.path("custom_id").asText());
        
        JsonNode response = root.path("response");
        if (!response.isMissingNode() && !response.isNull()) {
            int statusCode = response.path("status_code").asInt();
            JsonNode body = response.path("body");
            if (statusCode >= 200 && statusCode < 300) {
                ChatCompletion chatCompletion = ObjectMappers.jsonMapper().treeToValue(body, ChatCompletion.class);
                return new BatchOutputLine(index, BatchItemResult.success(toChatResponse(chatCompletion)));
            }
            return new BatchOutputLine(index, BatchItemResult.failure(toBatchError(statusCode, body)));
        }
        
        return new BatchOutputLine(index, BatchItemResult.failure(toBatchError(root.path("error"))));
    }
    
    static BatchState toBatchState(Batch.Status status) {
        return switch (status.value()) {
            case VALIDATING -> BatchState.PENDING;
            case IN_PROGRESS, FINALIZING, CANCELLING -> BatchState.RUNNING;
            case COMPLETED -> BatchState.SUCCEEDED;
            case FAILED -> BatchState.FAILED;
            case CANCELLED -> BatchState.CANCELLED;
            case EXPIRED -> BatchState.EXPIRED;
            case _UNKNOWN -> BatchState.UNSPECIFIED;
        };
    }
    
    private static int customIdIndex(String customId) {
        if (customId != null && customId.startsWith(CUSTOM_ID_PREFIX)) {
            return Integer.parseInt(customId.substring(CUSTOM_ID_PREFIX.length()));
        }
        throw new IllegalArgumentException("Unexpected OpenAI batch custom_id: " + customId);
    }
    
    @SuppressWarnings("deprecation")
    private static ChatResponse toChatResponse(ChatCompletion chatCompletion) {
        OpenAiOfficialChatResponseMetadata.Builder responseMetadataBuilder = OpenAiOfficialChatResponseMetadata.builder()
            .id(chatCompletion.id())
            .modelName(chatCompletion.model())
            .created(chatCompletion.created());
        
        if (!chatCompletion.choices().isEmpty()) {
            ChatCompletion.Choice choice = chatCompletion.choices().get(0);
            responseMetadataBuilder.finishReason(finishReasonFrom(choice.finishReason()));
            
            if (choice.message().toolCalls().isPresent()
                && choice.finishReason().equals(ChatCompletion.Choice.FinishReason.STOP)) {
                responseMetadataBuilder.finishReason(FinishReason.TOOL_EXECUTION);
            }
        }
        chatCompletion.usage().ifPresent(usage -> responseMetadataBuilder.tokenUsage(tokenUsageFrom(usage)));
        chatCompletion.serviceTier().ifPresent(serviceTier -> responseMetadataBuilder.serviceTier(serviceTier.toString()));
        chatCompletion.systemFingerprint().ifPresent(responseMetadataBuilder::systemFingerprint);
        
        return ChatResponse.builder()
            .aiMessage(aiMessageFrom(chatCompletion))
            .metadata(responseMetadataBuilder.build())
            .build();
    }
    
    private static BatchError toBatchError(com.openai.models.batches.BatchError error) {
        String code = error.code().orElse("0");
        List<Map<String, Object>> details = new ArrayList<>();
        details.add(Map.of("code", code));
        error.line().ifPresent(line -> details.add(Map.of("line", line)));
        error.param().ifPresent(param -> details.add(Map.of("param", param)));
        return new BatchError(parseCode(code), error.message().orElse("OpenAI batch request failed"), details);
    }
    
    private static BatchError toBatchError(JsonNode error) {
        String code = error.path("code").asText("0");
        String message = error.path("message").asText("OpenAI batch request failed");
        return new BatchError(parseCode(code), message, List.of(Map.of("code", code)));
    }
    
    private static BatchError toBatchError(int statusCode, JsonNode body) {
        JsonNode error = body.path("error");
        if (!error.isMissingNode() && !error.isNull()) {
            String code = error.path("code").asText(String.valueOf(statusCode));
            String message = error.path("message").asText("OpenAI batch request failed");
            return new BatchError(statusCode, message, List.of(Map.of("code", code)));
        }
        return new BatchError(statusCode, "OpenAI batch request failed", List.of());
    }
    
    private static int parseCode(String code) {
        try {
            return Integer.parseInt(code);
        }
        catch (NumberFormatException ignored) {
            return 0;
        }
    }
    
    static Map<String, Object> batchInputLine(String customId, JsonNode body) {
        Map<String, Object> line = new LinkedHashMap<>();
        line.put("custom_id", customId);
        line.put("method", "POST");
        line.put("url", CHAT_COMPLETIONS_BATCH_ENDPOINT);
        line.put("body", body);
        return line;
    }
    
    private void initWithoutCreatingClient(Builder builder) {
        ChatRequestParameters commonParameters;
        if (builder.defaultRequestParameters != null) {
            validate(builder.defaultRequestParameters);
            commonParameters = builder.defaultRequestParameters;
        }
        else {
            commonParameters = DefaultChatRequestParameters.EMPTY;
        }
        
        OpenAiOfficialChatRequestParameters openAiParameters;
        if (builder.defaultRequestParameters instanceof OpenAiOfficialChatRequestParameters openAiChatRequestParameters) {
            openAiParameters = openAiChatRequestParameters;
        }
        else {
            openAiParameters = OpenAiOfficialChatRequestParameters.EMPTY;
        }
        
        this.defaultRequestParameters = OpenAiOfficialChatRequestParameters.builder()
            .modelName(getOrDefault(builder.modelName, commonParameters.modelName()))
            .temperature(getOrDefault(builder.temperature, commonParameters.temperature()))
            .topP(getOrDefault(builder.topP, commonParameters.topP()))
            .frequencyPenalty(getOrDefault(builder.frequencyPenalty, commonParameters.frequencyPenalty()))
            .presencePenalty(getOrDefault(builder.presencePenalty, commonParameters.presencePenalty()))
            .maxOutputTokens(getOrDefault(builder.maxCompletionTokens, commonParameters.maxOutputTokens()))
            .stopSequences(getOrDefault(builder.stop, commonParameters.stopSequences()))
            .toolSpecifications(commonParameters.toolSpecifications())
            .toolChoice(commonParameters.toolChoice())
            .responseFormat(getOrDefault(fromOpenAiResponseFormat(builder.responseFormat), commonParameters.responseFormat()))
            .maxCompletionTokens(getOrDefault(builder.maxCompletionTokens, openAiParameters.maxCompletionTokens()))
            .logitBias(getOrDefault(builder.logitBias, openAiParameters.logitBias()))
            .parallelToolCalls(getOrDefault(builder.parallelToolCalls, openAiParameters.parallelToolCalls()))
            .seed(getOrDefault(builder.seed, openAiParameters.seed()))
            .user(getOrDefault(builder.user, openAiParameters.user()))
            .store(getOrDefault(builder.store, openAiParameters.store()))
            .metadata(getOrDefault(builder.metadata, openAiParameters.metadata()))
            .serviceTier(getOrDefault(builder.serviceTier, openAiParameters.serviceTier()))
            .reasoningEffort(openAiParameters.reasoningEffort())
            .build();
        
        this.modelProvider = detectModelProvider(
            builder.isMicrosoftFoundry,
            builder.isGitHubModels,
            builder.baseUrl,
            builder.microsoftFoundryDeploymentName,
            builder.azureOpenAIServiceVersion);
        this.responseFormat = builder.responseFormat;
        this.strictJsonSchema = getOrDefault(builder.strictJsonSchema, false);
        this.strictTools = getOrDefault(builder.strictTools, false);
        this.tokenCountEstimator = builder.tokenCountEstimator;
        this.listeners = copy(builder.listeners);
        this.supportedCapabilities = copy(builder.capabilities);
    }
    
    record BatchOutputLine(int index, BatchItemResult<ChatResponse> result) {
    }
    
    public static class Builder {
        
        private String                    baseUrl;
        private String                    apiKey;
        private Credential                credential;
        private String                    microsoftFoundryDeploymentName;
        private AzureOpenAIServiceVersion azureOpenAIServiceVersion;
        private String                    organizationId;
        private boolean                   isMicrosoftFoundry;
        private boolean                   isGitHubModels;
        private OpenAIClient              openAIClient;
        
        private ChatRequestParameters     defaultRequestParameters;
        private String                    modelName;
        private Double                    temperature;
        private Double                    topP;
        private List<String>              stop;
        private Integer                   maxCompletionTokens;
        private Double                    presencePenalty;
        private Double                    frequencyPenalty;
        private Map<String, Integer>      logitBias;
        private String                    responseFormat;
        private Boolean                   strictJsonSchema;
        private Integer                   seed;
        private String                    user;
        private Boolean                   strictTools;
        private Boolean                   parallelToolCalls;
        private Boolean                   store;
        private Map<String, String>       metadata;
        private String                    serviceTier;
        
        private Duration                  timeout;
        private Integer                   maxRetries;
        private Proxy                     proxy;
        private TokenCountEstimator       tokenCountEstimator;
        private Map<String, String>       customHeaders;
        private List<ChatModelListener>   listeners;
        private Set<Capability>           capabilities;
        
        public Builder defaultRequestParameters(ChatRequestParameters parameters) {
            this.defaultRequestParameters = parameters;
            return this;
        }
        
        public Builder modelName(String modelName) {
            this.modelName = modelName;
            return this;
        }
        
        public Builder modelName(com.openai.models.ChatModel modelName) {
            this.modelName = modelName.toString();
            return this;
        }
        
        public Builder baseUrl(String baseUrl) {
            this.baseUrl = baseUrl;
            return this;
        }
        
        public Builder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }
        
        public Builder credential(Credential credential) {
            this.credential = credential;
            return this;
        }
        
        /**
         * @deprecated Use {@link #microsoftFoundryDeploymentName(String)} instead.
         */
        @Deprecated
        public Builder azureDeploymentName(String azureDeploymentName) {
            this.microsoftFoundryDeploymentName = azureDeploymentName;
            return this;
        }
        
        public Builder microsoftFoundryDeploymentName(String microsoftFoundryDeploymentName) {
            this.microsoftFoundryDeploymentName = microsoftFoundryDeploymentName;
            return this;
        }
        
        public Builder azureOpenAIServiceVersion(AzureOpenAIServiceVersion azureOpenAIServiceVersion) {
            this.azureOpenAIServiceVersion = azureOpenAIServiceVersion;
            return this;
        }
        
        public Builder organizationId(String organizationId) {
            this.organizationId = organizationId;
            return this;
        }
        
        public Builder isGitHubModels(boolean isGitHubModels) {
            this.isGitHubModels = isGitHubModels;
            return this;
        }
        
        /**
         * @deprecated Use {@link #isMicrosoftFoundry(boolean)} instead.
         */
        @Deprecated
        public Builder isAzure(boolean isAzure) {
            this.isMicrosoftFoundry = isAzure;
            return this;
        }
        
        public Builder isMicrosoftFoundry(boolean isMicrosoftFoundry) {
            this.isMicrosoftFoundry = isMicrosoftFoundry;
            return this;
        }
        
        public Builder openAIClient(OpenAIClient openAIClient) {
            this.openAIClient = openAIClient;
            return this;
        }
        
        public Builder temperature(Double temperature) {
            this.temperature = temperature;
            return this;
        }
        
        public Builder topP(Double topP) {
            this.topP = topP;
            return this;
        }
        
        public Builder stop(List<String> stop) {
            this.stop = stop;
            return this;
        }
        
        public Builder maxCompletionTokens(Integer maxCompletionTokens) {
            this.maxCompletionTokens = maxCompletionTokens;
            return this;
        }
        
        public Builder presencePenalty(Double presencePenalty) {
            this.presencePenalty = presencePenalty;
            return this;
        }
        
        public Builder frequencyPenalty(Double frequencyPenalty) {
            this.frequencyPenalty = frequencyPenalty;
            return this;
        }
        
        public Builder logitBias(Map<String, Integer> logitBias) {
            this.logitBias = logitBias;
            return this;
        }
        
        public Builder responseFormat(String responseFormat) {
            this.responseFormat = responseFormat;
            return this;
        }
        
        public Builder strictJsonSchema(Boolean strictJsonSchema) {
            this.strictJsonSchema = strictJsonSchema;
            return this;
        }
        
        public Builder seed(Integer seed) {
            this.seed = seed;
            return this;
        }
        
        public Builder user(String user) {
            this.user = user;
            return this;
        }
        
        public Builder strictTools(Boolean strictTools) {
            this.strictTools = strictTools;
            return this;
        }
        
        public Builder parallelToolCalls(Boolean parallelToolCalls) {
            this.parallelToolCalls = parallelToolCalls;
            return this;
        }
        
        public Builder store(Boolean store) {
            this.store = store;
            return this;
        }
        
        public Builder metadata(Map<String, String> metadata) {
            this.metadata = metadata;
            return this;
        }
        
        public Builder serviceTier(String serviceTier) {
            this.serviceTier = serviceTier;
            return this;
        }
        
        public Builder timeout(Duration timeout) {
            this.timeout = timeout;
            return this;
        }
        
        public Builder maxRetries(Integer maxRetries) {
            this.maxRetries = maxRetries;
            return this;
        }
        
        public Builder proxy(Proxy proxy) {
            this.proxy = proxy;
            return this;
        }
        
        public Builder tokenCountEstimator(TokenCountEstimator tokenCountEstimator) {
            this.tokenCountEstimator = tokenCountEstimator;
            return this;
        }
        
        public Builder customHeaders(Map<String, String> customHeaders) {
            this.customHeaders = customHeaders;
            return this;
        }
        
        public Builder listeners(List<ChatModelListener> listeners) {
            this.listeners = listeners;
            return this;
        }
        
        public Builder supportedCapabilities(Set<Capability> capabilities) {
            this.capabilities = capabilities;
            return this;
        }
        
        public OpenAiOfficialBatchChatModel build() {
            return new OpenAiOfficialBatchChatModel(this);
        }
    }
}
