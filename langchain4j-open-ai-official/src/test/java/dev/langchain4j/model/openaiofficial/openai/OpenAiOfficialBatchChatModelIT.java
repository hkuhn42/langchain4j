package dev.langchain4j.model.openaiofficial.openai;

import static dev.langchain4j.model.batch.BatchState.PENDING;
import static dev.langchain4j.model.batch.BatchState.RUNNING;
import static dev.langchain4j.model.batch.BatchState.SUCCEEDED;
import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import java.util.stream.Stream;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.batch.BatchRequest;
import dev.langchain4j.model.batch.BatchResponse;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.openaiofficial.OpenAiOfficialBatchChatModel;

@EnabledIfEnvironmentVariable(named = "OPENAI_BASE_URL", matches = ".+")
@EnabledIfEnvironmentVariable(named = "OPENAI_MODEL_NAME", matches = ".+")
@EnabledIfEnvironmentVariable(named = "OPENAI_API_KEY", matches = ".+")
class OpenAiOfficialBatchChatModelIT {
    
    @Test
    void submitBatch() {
        
        OpenAiOfficialBatchChatModel model = OpenAiOfficialBatchChatModel.builder()
            .baseUrl(System.getenv("OPENAI_BASE_URL"))
            .modelName(System.getenv("OPENAI_MODEL_NAME"))
            .apiKey(System.getenv("OPENAI_API_KEY"))
            .build();
        
        BatchResponse<ChatResponse> response = model.submit(new BatchRequest<>(List.of(ChatRequest.builder()
            .messages(UserMessage.from("Return only the word OK."))
            .build())));
        System.out.println(response.batchId());
        assertThat(response.batchId()).isNotBlank();
        assertThat(response.state()).isIn(PENDING, RUNNING, SUCCEEDED);
    }
    
    @ParameterizedTest
    @MethodSource("batchIds")
    @EnabledIfEnvironmentVariable(named = "OPENAI_BATCH_ID", matches = ".+")
    void checkBatch(String batchId) {
        Assumptions.assumeTrue(batchId != null && !batchId.isBlank(), "batchId is not set");
        
        OpenAiOfficialBatchChatModel model = OpenAiOfficialBatchChatModel.builder()
            .baseUrl(System.getenv("OPENAI_BASE_URL"))
            .modelName(System.getenv("OPENAI_MODEL_NAME"))
            .apiKey(System.getenv("OPENAI_API_KEY"))
            .build();
        
        BatchResponse<ChatResponse> response = model.retrieve(batchId);
        
        System.out.println("batchId=" + response.batchId());
        System.out.println("status=" + response.state());
        
        if (response.state().isTerminal()) {
            System.out.println("responses=" + response.responses());
            System.out.println("errors=" + response.errors());
        }
        
        assertThat(response.batchId()).isEqualTo(batchId);
    }
    
    static Stream<String> batchIds() {
        return Stream.of(System.getenv("OPENAI_BATCH_ID"));
    }
    
}
