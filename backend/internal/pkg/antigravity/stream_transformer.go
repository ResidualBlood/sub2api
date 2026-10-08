package antigravity

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"strings"
)

// BlockType 内容块类型
type BlockType int

const (
	BlockTypeNone BlockType = iota
	BlockTypeText
	BlockTypeThinking
	BlockTypeFunction
)

// UsageMapHook is a callback that can modify usage data before it's emitted in SSE events.
type UsageMapHook func(usageMap map[string]any)

// StreamingProcessor 流式响应处理器
type StreamingProcessor struct {
	blockType         BlockType
	blockIndex        int
	messageStartSent  bool
	messageStopSent   bool
	usedTool          bool
	pendingSignature  string
	trailingSignature string
	originalModel     string
	webSearchQueries  []string
	groundingChunks   []GeminiGroundingChunk
	usageMapHook      UsageMapHook

	// 累计 usage
	inputTokens       int
	outputTokens      int
	cacheReadTokens   int
	imageOutputTokens int
	hasContent        bool

	// textCallBuffer 用于跨 chunk 缓冲与识别模型在文本正文中输出的伪工具调用
	// 例如: call:default_api:bash{command: "ls"}
	textCallBuffer string
}

// NewStreamingProcessor 创建流式响应处理器
func NewStreamingProcessor(originalModel string) *StreamingProcessor {
	return &StreamingProcessor{
		blockType:     BlockTypeNone,
		originalModel: originalModel,
	}
}

// SetUsageMapHook sets an optional hook that modifies usage maps before they are emitted.
func (p *StreamingProcessor) SetUsageMapHook(fn UsageMapHook) {
	p.usageMapHook = fn
}

func usageToMap(u ClaudeUsage) map[string]any {
	m := map[string]any{
		"input_tokens":  u.InputTokens,
		"output_tokens": u.OutputTokens,
	}
	if u.CacheCreationInputTokens > 0 {
		m["cache_creation_input_tokens"] = u.CacheCreationInputTokens
	}
	if u.CacheReadInputTokens > 0 {
		m["cache_read_input_tokens"] = u.CacheReadInputTokens
	}
	if u.ImageOutputTokens > 0 {
		m["image_output_tokens"] = u.ImageOutputTokens
	}
	return m
}

// ProcessLine 处理 SSE 行，返回 Claude SSE 事件
func (p *StreamingProcessor) ProcessLine(line string) []byte {
	line = strings.TrimSpace(line)
	if line == "" || !strings.HasPrefix(line, "data:") {
		return nil
	}

	data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
	if data == "" || data == "[DONE]" {
		return nil
	}

	// 解包 v1internal 响应
	var v1Resp V1InternalResponse
	if err := json.Unmarshal([]byte(data), &v1Resp); err != nil {
		// 尝试直接解析为 GeminiResponse
		var directResp GeminiResponse
		if err2 := json.Unmarshal([]byte(data), &directResp); err2 != nil {
			return nil
		}
		v1Resp.Response = directResp
		v1Resp.ResponseID = directResp.ResponseID
		v1Resp.ModelVersion = directResp.ModelVersion
	}

	geminiResp := &v1Resp.Response

	var result bytes.Buffer

	// 发送 message_start
	if !p.messageStartSent {
		_, _ = result.Write(p.emitMessageStart(&v1Resp))
	}

	// 更新 usage
	// 注意：Gemini 的 promptTokenCount 包含 cachedContentTokenCount，
	// 但 Claude 的 input_tokens 不包含 cache_read_input_tokens，需要减去
	if geminiResp.UsageMetadata != nil {
		cached := geminiResp.UsageMetadata.CachedContentTokenCount
		p.inputTokens = geminiResp.UsageMetadata.PromptTokenCount - cached
		p.outputTokens = geminiResp.UsageMetadata.CandidatesTokenCount + geminiResp.UsageMetadata.ThoughtsTokenCount
		p.cacheReadTokens = cached
		p.imageOutputTokens = geminiResp.UsageMetadata.ImageOutputTokens()
	}

	// 处理 parts
	if len(geminiResp.Candidates) > 0 && geminiResp.Candidates[0].Content != nil {
		for _, part := range geminiResp.Candidates[0].Content.Parts {
			_, _ = result.Write(p.processPart(&part))
		}
	}

	if len(geminiResp.Candidates) > 0 {
		p.captureGrounding(geminiResp.Candidates[0].GroundingMetadata)
	}

	// 检查是否结束
	if len(geminiResp.Candidates) > 0 {
		finishReason := geminiResp.Candidates[0].FinishReason
		if finishReason == "MALFORMED_FUNCTION_CALL" {
			log.Printf("[Antigravity] MALFORMED_FUNCTION_CALL detected in stream for model %s", p.originalModel)
			if geminiResp.Candidates[0].Content != nil {
				if b, err := json.Marshal(geminiResp.Candidates[0].Content); err == nil {
					log.Printf("[Antigravity] Malformed content: %s", string(b))
				}
			}
		}
		if finishReason != "" {
			_, _ = result.Write(p.emitFinish(finishReason))
		}
	}

	return result.Bytes()
}

// Finish 结束处理，返回最终事件和用量。
// 若整个流未收到任何可解析的上游数据（messageStartSent == false），
// 则不补发任何结束事件，防止客户端收到没有 message_start 的残缺流。
func (p *StreamingProcessor) Finish() ([]byte, *ClaudeUsage) {
	usage := &ClaudeUsage{
		InputTokens:          p.inputTokens,
		OutputTokens:         p.outputTokens,
		CacheReadInputTokens: p.cacheReadTokens,
		ImageOutputTokens:    p.imageOutputTokens,
	}

	if !p.messageStartSent {
		return nil, usage
	}

	var result bytes.Buffer

	// 如果 textCallBuffer 还有未排出的剩余文本（不是合法的伪调用），排出为普通文本
	if p.textCallBuffer != "" {
		remaining := p.textCallBuffer
		p.textCallBuffer = ""
		_, _ = result.Write(p.emitPlainText(remaining))
	}

	if !p.messageStopSent {
		_, _ = result.Write(p.emitFinish(""))
	}

	return result.Bytes(), usage
}

// MessageStartSent 报告流中是否已发出过 message_start 事件（即是否收到过有效的上游数据）
func (p *StreamingProcessor) MessageStartSent() bool {
	return p.messageStartSent
}

// HasContent reports whether any substantive text, thinking, or tool calls were emitted.
func (p *StreamingProcessor) HasContent() bool {
	return p.hasContent
}

// emitMessageStart 发送 message_start 事件
func (p *StreamingProcessor) emitMessageStart(v1Resp *V1InternalResponse) []byte {
	if p.messageStartSent {
		return nil
	}

	usage := ClaudeUsage{}
	if v1Resp.Response.UsageMetadata != nil {
		cached := v1Resp.Response.UsageMetadata.CachedContentTokenCount
		usage.InputTokens = v1Resp.Response.UsageMetadata.PromptTokenCount - cached
		usage.OutputTokens = v1Resp.Response.UsageMetadata.CandidatesTokenCount + v1Resp.Response.UsageMetadata.ThoughtsTokenCount
		usage.CacheReadInputTokens = cached
		usage.ImageOutputTokens = v1Resp.Response.UsageMetadata.ImageOutputTokens()
	}

	responseID := v1Resp.ResponseID
	if responseID == "" {
		responseID = v1Resp.Response.ResponseID
	}
	if responseID == "" {
		responseID = generateAnthropicMsgID()
	}

	var usageValue any = usage
	if p.usageMapHook != nil {
		usageMap := usageToMap(usage)
		p.usageMapHook(usageMap)
		usageValue = usageMap
	}

	message := map[string]any{
		"id":            responseID,
		"type":          "message",
		"role":          "assistant",
		"content":       []any{},
		"model":         p.originalModel,
		"stop_reason":   nil,
		"stop_sequence": nil,
		"usage":         usageValue,
	}

	event := map[string]any{
		"type":    "message_start",
		"message": message,
	}

	p.messageStartSent = true
	return p.formatSSE("message_start", event)
}

// processPart 处理单个 part
func (p *StreamingProcessor) processPart(part *GeminiPart) []byte {
	var result bytes.Buffer
	signature := part.ThoughtSignature

	// 1. FunctionCall 处理
	if part.FunctionCall != nil {
		// 先处理 trailingSignature
		if p.trailingSignature != "" {
			_, _ = result.Write(p.endBlock())
			_, _ = result.Write(p.emitEmptyThinkingWithSignature(p.trailingSignature))
			p.trailingSignature = ""
		}

		_, _ = result.Write(p.processFunctionCall(part.FunctionCall, signature))
		return result.Bytes()
	}

	// 2. Text 处理
	if part.Text != "" || part.Thought {
		if part.Thought {
			_, _ = result.Write(p.processThinking(part.Text, signature))
		} else {
			_, _ = result.Write(p.processText(part.Text, signature))
		}
	}

	// 3. InlineData (Image) 处理
	if part.InlineData != nil && part.InlineData.Data != "" {
		markdownImg := fmt.Sprintf("![image](data:%s;base64,%s)",
			part.InlineData.MimeType, part.InlineData.Data)
		_, _ = result.Write(p.processText(markdownImg, ""))
	}

	return result.Bytes()
}

func (p *StreamingProcessor) captureGrounding(grounding *GeminiGroundingMetadata) {
	if grounding == nil {
		return
	}

	if len(grounding.WebSearchQueries) > 0 && len(p.webSearchQueries) == 0 {
		p.webSearchQueries = append([]string(nil), grounding.WebSearchQueries...)
	}

	if len(grounding.GroundingChunks) > 0 && len(p.groundingChunks) == 0 {
		p.groundingChunks = append([]GeminiGroundingChunk(nil), grounding.GroundingChunks...)
	}
}

// processThinking 处理 thinking
func (p *StreamingProcessor) processThinking(text, signature string) []byte {
	var result bytes.Buffer

	// 处理之前的 trailingSignature
	if p.trailingSignature != "" {
		_, _ = result.Write(p.endBlock())
		_, _ = result.Write(p.emitEmptyThinkingWithSignature(p.trailingSignature))
		p.trailingSignature = ""
	}

	// 开始或继续 thinking 块
	if p.blockType != BlockTypeThinking {
		_, _ = result.Write(p.startBlock(BlockTypeThinking, map[string]any{
			"type":      "thinking",
			"thinking":  "",
			"signature": "",
		}))
	}

	if text != "" {
		p.hasContent = true
		_, _ = result.Write(p.emitDelta("thinking_delta", map[string]any{
			"thinking": text,
		}))
	}

	// 暂存签名
	if signature != "" {
		p.pendingSignature = signature
	}

	return result.Bytes()
}

// processText 处理普通 text
func (p *StreamingProcessor) processText(text, signature string) []byte {
	// 空 text 带签名 - 暂存
	if text == "" {
		if signature != "" {
			p.trailingSignature = signature
		}
		return nil
	}

	// 签名非空时直接走标准文本逻辑（不参与伪调用缓存）
	if signature != "" {
		var result bytes.Buffer
		if p.textCallBuffer != "" {
			prev := p.textCallBuffer
			p.textCallBuffer = ""
			_, _ = result.Write(p.emitPlainText(prev))
		}
		_, _ = result.Write(p.emitTextWithSignature(text, signature))
		return result.Bytes()
	}

	return p.processBufferedText(text)
}

func (p *StreamingProcessor) emitTextWithSignature(text, signature string) []byte {
	var result bytes.Buffer
	if p.trailingSignature != "" {
		_, _ = result.Write(p.endBlock())
		_, _ = result.Write(p.emitEmptyThinkingWithSignature(p.trailingSignature))
		p.trailingSignature = ""
	}
	_, _ = result.Write(p.startBlock(BlockTypeText, map[string]any{
		"type": "text",
		"text": "",
	}))
	_, _ = result.Write(p.emitDelta("text_delta", map[string]any{
		"text": text,
	}))
	_, _ = result.Write(p.endBlock())
	_, _ = result.Write(p.emitEmptyThinkingWithSignature(signature))
	return result.Bytes()
}

func (p *StreamingProcessor) emitPlainText(text string) []byte {
	if text == "" {
		return nil
	}
	var result bytes.Buffer
	if p.trailingSignature != "" {
		_, _ = result.Write(p.endBlock())
		_, _ = result.Write(p.emitEmptyThinkingWithSignature(p.trailingSignature))
		p.trailingSignature = ""
	}
	if p.blockType != BlockTypeText {
		_, _ = result.Write(p.startBlock(BlockTypeText, map[string]any{
			"type": "text",
			"text": "",
		}))
	}
	p.hasContent = true
	_, _ = result.Write(p.emitDelta("text_delta", map[string]any{
		"text": text,
	}))
	return result.Bytes()
}

const (
	geminiCallMarker = "call:default_api:"
	maxHeldCallLen   = 256 << 10
)

// processBufferedText 维护状态机，检测并抢救文本中裸输出的伪工具调用
func (p *StreamingProcessor) processBufferedText(text string) []byte {
	s := p.textCallBuffer + text
	p.textCallBuffer = ""

	var result bytes.Buffer

	for s != "" {
		idx := strings.Index(s, geminiCallMarker)
		if idx < 0 {
			// 未包含完整 marker，但末尾可能只是 marker 的前缀片段，做尾部暂存
			k := len(geminiCallMarker) - 1
			for ; k > 0 && !strings.HasSuffix(s, geminiCallMarker[:k]); k-- {
			}
			emitPart := s[:len(s)-k]
			_, _ = result.Write(p.emitPlainText(emitPart))
			p.textCallBuffer = s[len(s)-k:]
			return result.Bytes()
		}

		// marker 前面的普通文本直接发出
		if idx > 0 {
			_, _ = result.Write(p.emitPlainText(s[:idx]))
		}
		s = s[idx:]

		callName, args, consumed := parsePseudoTextCall(s)
		switch {
		case consumed == 0 && len(s) < maxHeldCallLen:
			// 疑似伪工具调用但尚未接收完整，暂存至缓冲区等待后续 chunk
			p.textCallBuffer = s
			return result.Bytes()
		case consumed <= 0:
			// 不是合法伪调用，当普通文本吐出 marker 首段，继续后移
			_, _ = result.Write(p.emitPlainText(s[:len(geminiCallMarker)]))
			s = s[len(geminiCallMarker):]
		default:
			// 成功识别出伪工具调用，转换为标准的 Claude tool_use 事件下发
			if p.blockType != BlockTypeNone {
				_, _ = result.Write(p.endBlock())
			}
			fc := &GeminiFunctionCall{
				Name: callName,
				Args: args,
				ID:   fmt.Sprintf("call_%s", generateRandomID()),
			}
			_, _ = result.Write(p.processFunctionCall(fc, ""))
			s = s[consumed:]
		}
	}

	return result.Bytes()
}

func isCallIdent(c byte) bool {
	return c == '_' || c == '-' || c == '.' || ('0' <= c && c <= '9') || ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z')
}

// parsePseudoTextCall 解析 call:default_api:<tool>{<args>} 格式
func parsePseudoTextCall(s string) (string, map[string]any, int) {
	j := len(geminiCallMarker)
	for j < len(s) && isCallIdent(s[j]) {
		j++
	}
	if j == len(s) {
		return "", nil, 0
	}
	name := s[len(geminiCallMarker):j]
	if name == "" || s[j] != '{' {
		return "", nil, -1
	}

	end := findClosingBrace(s, j)
	if end < 0 {
		return "", nil, 0
	}

	argsRaw := strings.TrimSpace(s[j : end+1])
	var args map[string]any
	if err := json.Unmarshal([]byte(argsRaw), &args); err != nil {
		// 容错: 某些伪调用可能没有引号或参数异常，放宽解析
		return "", nil, -1
	}

	return name, args, end + 1
}

func findClosingBrace(s string, start int) int {
	depth := 0
	inString := false
	var escape bool

	for i := start; i < len(s); i++ {
		c := s[i]
		if escape {
			escape = false
			continue
		}
		if c == '\\' {
			escape = true
			continue
		}
		if c == '"' {
			inString = !inString
			continue
		}
		if inString {
			continue
		}
		if c == '{' {
			depth++
		} else if c == '}' {
			depth--
			if depth == 0 {
				return i
			}
		}
	}
	return -1
}

// processFunctionCall 处理 function call
func (p *StreamingProcessor) processFunctionCall(fc *GeminiFunctionCall, signature string) []byte {
	var result bytes.Buffer

	p.usedTool = true
	p.hasContent = true

	toolID := fc.ID
	if toolID == "" {
		toolID = fmt.Sprintf("%s-%s", fc.Name, generateRandomID())
	}

	toolUse := map[string]any{
		"type":  "tool_use",
		"id":    toolID,
		"name":  fc.Name,
		"input": map[string]any{},
	}

	if signature != "" {
		toolUse["signature"] = signature
	}

	_, _ = result.Write(p.startBlock(BlockTypeFunction, toolUse))

	// 发送 input_json_delta
	if fc.Args != nil {
		argsJSON, _ := json.Marshal(fc.Args)
		_, _ = result.Write(p.emitDelta("input_json_delta", map[string]any{
			"partial_json": string(argsJSON),
		}))
	}

	_, _ = result.Write(p.endBlock())

	return result.Bytes()
}

// startBlock 开始新的内容块
func (p *StreamingProcessor) startBlock(blockType BlockType, contentBlock map[string]any) []byte {
	var result bytes.Buffer

	if p.blockType != BlockTypeNone {
		_, _ = result.Write(p.endBlock())
	}

	event := map[string]any{
		"type":          "content_block_start",
		"index":         p.blockIndex,
		"content_block": contentBlock,
	}

	_, _ = result.Write(p.formatSSE("content_block_start", event))
	p.blockType = blockType

	return result.Bytes()
}

// endBlock 结束当前内容块
func (p *StreamingProcessor) endBlock() []byte {
	if p.blockType == BlockTypeNone {
		return nil
	}

	var result bytes.Buffer

	// Thinking 块结束时发送暂存的签名
	if p.blockType == BlockTypeThinking && p.pendingSignature != "" {
		_, _ = result.Write(p.emitDelta("signature_delta", map[string]any{
			"signature": p.pendingSignature,
		}))
		p.pendingSignature = ""
	}

	event := map[string]any{
		"type":  "content_block_stop",
		"index": p.blockIndex,
	}

	_, _ = result.Write(p.formatSSE("content_block_stop", event))

	p.blockIndex++
	p.blockType = BlockTypeNone

	return result.Bytes()
}

// emitDelta 发送 delta 事件
func (p *StreamingProcessor) emitDelta(deltaType string, deltaContent map[string]any) []byte {
	delta := map[string]any{
		"type": deltaType,
	}
	for k, v := range deltaContent {
		delta[k] = v
	}

	event := map[string]any{
		"type":  "content_block_delta",
		"index": p.blockIndex,
		"delta": delta,
	}

	return p.formatSSE("content_block_delta", event)
}

// emitEmptyThinkingWithSignature 发送空 thinking 块承载签名
func (p *StreamingProcessor) emitEmptyThinkingWithSignature(signature string) []byte {
	var result bytes.Buffer

	_, _ = result.Write(p.startBlock(BlockTypeThinking, map[string]any{
		"type":      "thinking",
		"thinking":  "",
		"signature": "",
	}))
	_, _ = result.Write(p.emitDelta("thinking_delta", map[string]any{
		"thinking": "",
	}))
	_, _ = result.Write(p.emitDelta("signature_delta", map[string]any{
		"signature": signature,
	}))
	_, _ = result.Write(p.endBlock())

	return result.Bytes()
}

// emitFinish 发送结束事件
func (p *StreamingProcessor) emitFinish(finishReason string) []byte {
	var result bytes.Buffer

	// 关闭最后一个块
	_, _ = result.Write(p.endBlock())

	// 处理 trailingSignature
	if p.trailingSignature != "" {
		_, _ = result.Write(p.emitEmptyThinkingWithSignature(p.trailingSignature))
		p.trailingSignature = ""
	}

	if len(p.webSearchQueries) > 0 || len(p.groundingChunks) > 0 {
		groundingText := buildGroundingText(&GeminiGroundingMetadata{
			WebSearchQueries: p.webSearchQueries,
			GroundingChunks:  p.groundingChunks,
		})
		if groundingText != "" {
			_, _ = result.Write(p.startBlock(BlockTypeText, map[string]any{
				"type": "text",
				"text": "",
			}))
			_, _ = result.Write(p.emitDelta("text_delta", map[string]any{
				"text": groundingText,
			}))
			_, _ = result.Write(p.endBlock())
		}
	}

	// 确定 stop_reason
	stopReason := "end_turn"
	if p.usedTool {
		stopReason = "tool_use"
	} else if finishReason == "MAX_TOKENS" {
		stopReason = "max_tokens"
	}

	usage := ClaudeUsage{
		InputTokens:          p.inputTokens,
		OutputTokens:         p.outputTokens,
		CacheReadInputTokens: p.cacheReadTokens,
		ImageOutputTokens:    p.imageOutputTokens,
	}

	var usageValue any = usage
	if p.usageMapHook != nil {
		usageMap := usageToMap(usage)
		p.usageMapHook(usageMap)
		usageValue = usageMap
	}

	deltaEvent := map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   stopReason,
			"stop_sequence": nil,
		},
		"usage": usageValue,
	}

	_, _ = result.Write(p.formatSSE("message_delta", deltaEvent))

	if !p.messageStopSent {
		stopEvent := map[string]any{
			"type": "message_stop",
		}
		_, _ = result.Write(p.formatSSE("message_stop", stopEvent))
		p.messageStopSent = true
	}

	return result.Bytes()
}

// formatSSE 格式化 SSE 事件
func (p *StreamingProcessor) formatSSE(eventType string, data any) []byte {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return nil
	}

	return []byte(fmt.Sprintf("event: %s\ndata: %s\n\n", eventType, string(jsonData)))
}
