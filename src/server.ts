import { createWorkersAI } from "workers-ai-provider";
import { callable, routeAgentRequest, type Schedule } from "agents";
import { getSchedulePrompt, scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import { convertToModelMessages, stepCountIs, streamText, tool } from "ai";
import { z } from "zod";

export class ChatAgent extends AIChatAgent<Env> {
  maxPersistedMessages = 100;
  chatRecovery = true;

  // Do not wait indefinitely for MCP connections.
  waitForMcpConnections = { timeout: 10_000 };

  onStart() {
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: { "content-type": "text/html" },
            status: 200
          });
        }

        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          {
            headers: { "content-type": "text/plain" },
            status: 400
          }
        );
      }
    });
  }

  @callable()
  async addServer(name: string, url: string) {
    return await this.addMcpServer(name, url);
  }

  @callable()
  async removeServer(serverId: string) {
    await this.removeMcpServer(serverId);
  }

  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    console.log("Chat request received");

    try {
      const workersai = createWorkersAI({
        binding: this.env.AI
      });

      const currentMessages = await convertToModelMessages(this.messages);

      console.log("Messages converted:", currentMessages.length);

      // Load MCP tools after the connection wait.
      const mcpTools = this.mcp.getAITools();

      const result = streamText({
        model: workersai("@cf/zai-org/glm-4.7-flash"),

        system: `You are a helpful assistant.

You can answer general questions, help with calculations,
search the live web, check weather using the weather tool,
and schedule tasks.

WEB SEARCH RULES:
- For current facts, recent news, public figures, and information
  that may have changed, use webSearch before answering.
- Base factual claims about current information on the returned
  search results, not guesses or memory alone.
- Include a Sources section with clickable Markdown links.
- Use the exact title and URL returned by webSearch.
- Never invent, alter, or omit URLs when search results provide them.
- If the search fails or returns no results, say so clearly.
- Do not claim that you searched if the tool was not called.

When a user asks about an image, answer based on the image
content provided to you.

Current date and scheduling instructions:
${getSchedulePrompt({ date: new Date() })}

If the user asks to schedule a task, use the scheduleTask tool.
Do not claim a task was scheduled unless the tool succeeds.`,

        messages: currentMessages,

        tools: {
          // Tools from connected MCP servers.
          ...mcpTools,

          // Web search tool using Firecrawl.
          webSearch: tool({
            description:
              "Search the live web for current facts, recent news, public figures, and information that may have changed. Use this tool before answering current-information questions. Return actual source titles, URLs, and descriptions.",
            inputSchema: z.object({
              query: z.string().describe("The web search query")
            }),
            execute: async ({ query }) => {
              try {
                const response = await fetch(
                  "https://api.firecrawl.dev/v2/search",
                  {
                    method: "POST",
                    headers: {
                      "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                      query,
                      limit: 5
                    })
                  }
                );

                if (!response.ok) {
                  console.error("Web search failed:", response.status);

                  return {
                    error: `Web search failed with HTTP ${response.status}.`
                  };
                }

                const payload = (await response.json()) as {
                  success?: boolean;
                  data?: {
                    web?: Array<{
                      title?: string;
                      url?: string;
                      description?: string;
                    }>;
                  };
                };

                const results = (payload.data?.web ?? [])
                  .filter(
                    (item) =>
                      typeof item.url === "string" &&
                      item.url.startsWith("https://")
                  )
                  .map((item) => ({
                    title: item.title ?? "Untitled",
                    url: item.url!,
                    description: item.description ?? ""
                  }));

                console.log(`Web search returned ${results.length} results`);

                if (results.length === 0) {
                  return {
                    error: "Web search returned no usable results."
                  };
                }

                // Return a clearly structured object containing real URLs.
                return { results };
              } catch (error) {
                console.error("Web search request error:", error);

                return {
                  error: "Web search could not be completed."
                };
              }
            }
          }),

          // Server-side weather tool.
          getWeather: tool({
            description: "Get weather information for a city",
            inputSchema: z.object({
              city: z.string().describe("City name")
            }),
            execute: async ({ city }) => {
              // Demo data only; replace with a real weather API.
              const conditions = ["sunny", "cloudy", "rainy", "snowy"];
              const temperature = Math.floor(Math.random() * 30) + 5;

              return {
                city,
                temperature,
                condition:
                  conditions[Math.floor(Math.random() * conditions.length)],
                unit: "celsius",
                note: "Simulated weather data"
              };
            }
          }),

          // Client-side tool: the frontend must implement it.
          getUserTimezone: tool({
            description: "Get the user's timezone from their browser.",
            inputSchema: z.object({})
          }),

          // Calculator tool.
          calculate: tool({
            description:
              "Calculate an arithmetic expression using two numbers.",
            inputSchema: z.object({
              a: z.number().describe("First number"),
              b: z.number().describe("Second number"),
              operator: z.enum(["+", "-", "*", "/", "%"])
            }),

            execute: async ({ a, b, operator }) => {
              if (operator === "/" && b === 0) {
                return { error: "Division by zero" };
              }

              const operations: Record<
                string,
                (x: number, y: number) => number
              > = {
                "+": (x, y) => x + y,
                "-": (x, y) => x - y,
                "*": (x, y) => x * y,
                "/": (x, y) => x / y,
                "%": (x, y) => x % y
              };

              return {
                expression: `${a} ${operator} ${b}`,
                result: operations[operator](a, b)
              };
            }
          }),

          // Scheduling tool.
          scheduleTask: tool({
            description: "Schedule a task to run at a later time.",
            inputSchema: scheduleSchema,

            execute: async ({ when, description }) => {
              if (when.type === "no-schedule") {
                return "Not a valid schedule input";
              }

              const input =
                when.type === "scheduled"
                  ? when.date
                  : when.type === "delayed"
                    ? when.delayInSeconds
                    : when.type === "cron"
                      ? when.cron
                      : null;

              if (!input) {
                return "Invalid schedule type";
              }

              try {
                await this.schedule(input, "executeTask", description, {
                  idempotent: true
                });

                return `Task scheduled: "${description}"`;
              } catch (error) {
                console.error("Scheduling error:", error);

                return `Error scheduling task: ${
                  error instanceof Error ? error.message : String(error)
                }`;
              }
            }
          })
        },

        stopWhen: stepCountIs(20),

        abortSignal: options?.abortSignal,

        onError: ({ error }) => {
          console.error("AI stream error:", error);
        }
      });

      console.log("AI stream created");

      return result.toUIMessageStreamResponse();
    } catch (error) {
      console.error("Chat generation failed:", error);

      return new Response(
        `AI chat error: ${
          error instanceof Error ? error.message : String(error)
        }`,
        {
          status: 500,
          headers: {
            "Content-Type": "text/plain; charset=utf-8"
          }
        }
      );
    }
  }

  async executeTask(description: string, _task: Schedule<string>) {
    console.log(`Executing scheduled task: ${description}`);

    this.broadcast(
      JSON.stringify({
        type: "scheduled-task",
        description,
        timestamp: new Date().toISOString()
      })
    );
  }
}

export default {
  async fetch(request: Request, env: Env, _ctx: ExecutionContext) {
    try {
      const agentResponse = await routeAgentRequest(request, env);

      if (agentResponse) {
        return agentResponse;
      }

      if (env.ASSETS) {
        return await env.ASSETS.fetch(request);
      }

      return new Response("Not found", {
        status: 404
      });
    } catch (error) {
      console.error("Worker request failed:", error);

      return new Response("Internal server error", {
        status: 500
      });
    }
  }
} satisfies ExportedHandler<Env>;
