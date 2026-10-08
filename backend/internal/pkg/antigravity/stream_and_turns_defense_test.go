package antigravity

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestTransformClaudeToGemini_SanitizeTurnOrder(t *testing.T) {
	t.Run("consecutive same role messages are merged", func(t *testing.T) {
		req := &ClaudeRequest{
			Model: "claude-3-5-sonnet-latest",
			Messages: []ClaudeMessage{
				{Role: "user", Content: json.RawMessage(`"hello 1"`)},
				{Role: "user", Content: json.RawMessage(`"hello 2"`)},
				{Role: "assistant", Content: json.RawMessage(`"reply 1"`)},
				{Role: "assistant", Content: json.RawMessage(`"reply 2"`)},
			},
		}

		raw, err := TransformClaudeToGemini(req, "test-proj", "gemini-2.5-pro")
		require.NoError(t, err)

		var v1Req V1InternalRequest
		require.NoError(t, json.Unmarshal(raw, &v1Req))

		// 应该合并为 1 个 user + 1 个 model，而不是 2 user + 2 model
		require.Len(t, v1Req.Request.Contents, 2)
		require.Equal(t, "user", v1Req.Request.Contents[0].Role)
		require.Len(t, v1Req.Request.Contents[0].Parts, 2)
		require.Equal(t, "hello 1", v1Req.Request.Contents[0].Parts[0].Text)
		require.Equal(t, "hello 2", v1Req.Request.Contents[0].Parts[1].Text)

		require.Equal(t, "model", v1Req.Request.Contents[1].Role)
		require.Len(t, v1Req.Request.Contents[1].Parts, 2)
		require.Equal(t, "reply 1", v1Req.Request.Contents[1].Parts[0].Text)
		require.Equal(t, "reply 2", v1Req.Request.Contents[1].Parts[1].Text)
	})

	t.Run("first turn model with multiple turns prepends user placeholder", func(t *testing.T) {
		req := &ClaudeRequest{
			Model: "claude-3-5-sonnet-latest",
			Messages: []ClaudeMessage{
				{Role: "assistant", Content: json.RawMessage(`"prior model thought"`)},
				{Role: "user", Content: json.RawMessage(`"what next?"`)},
			},
		}

		raw, err := TransformClaudeToGemini(req, "test-proj", "gemini-2.5-pro")
		require.NoError(t, err)

		var v1Req V1InternalRequest
		require.NoError(t, json.Unmarshal(raw, &v1Req))

		// 必须以 user 开头
		require.GreaterOrEqual(t, len(v1Req.Request.Contents), 3)
		require.Equal(t, "user", v1Req.Request.Contents[0].Role)
		require.Contains(t, v1Req.Request.Contents[0].Parts[0].Text, "Continuing from previous AI thoughts")
		require.Equal(t, "model", v1Req.Request.Contents[1].Role)
		require.Equal(t, "user", v1Req.Request.Contents[2].Role)
	})
}

func TestStreamingProcessor_RecoversPseudoToolCall(t *testing.T) {
	chunk1JSON, _ := json.Marshal(map[string]any{
		"response": map[string]any{
			"candidates": []any{
				map[string]any{
					"content": map[string]any{
						"parts": []any{
							map[string]any{"text": `Let me run a command. call:default_api:bash{"command":"ls`},
						},
					},
				},
			},
		},
	})
	chunk2JSON, _ := json.Marshal(map[string]any{
		"response": map[string]any{
			"candidates": []any{
				map[string]any{
					"content": map[string]any{
						"parts": []any{
							map[string]any{"text": ` -la"} and that's it.`},
						},
					},
				},
			},
		},
	})
	chunk3JSON, _ := json.Marshal(map[string]any{
		"response": map[string]any{
			"candidates": []any{
				map[string]any{
					"finishReason": "STOP",
				},
			},
		},
	})

	p := NewStreamingProcessor("claude-3-5-sonnet-latest")

	chunk1Line := "data: " + string(chunk1JSON)
	chunk2Line := "data: " + string(chunk2JSON)
	chunk3Line := "data: " + string(chunk3JSON)

	var rawLines []string
	if out := p.ProcessLine(chunk1Line); len(out) > 0 {
		rawLines = append(rawLines, string(out))
	}
	if out := p.ProcessLine(chunk2Line); len(out) > 0 {
		rawLines = append(rawLines, string(out))
	}
	if out := p.ProcessLine(chunk3Line); len(out) > 0 {
		rawLines = append(rawLines, string(out))
	}
	fin, _ := p.Finish()
	if len(fin) > 0 {
		rawLines = append(rawLines, string(fin))
	}

	combined := strings.Join(rawLines, "")
	t.Logf("combined output:\n%s", combined)

	require.Contains(t, combined, `Let me run a command. `)
	require.Contains(t, combined, `"type":"tool_use"`)
	require.Contains(t, combined, `"name":"bash"`)
	require.Contains(t, combined, `ls -la`)
	require.Contains(t, combined, ` and that's it.`)
}
