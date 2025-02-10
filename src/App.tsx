import { useRef, useState } from "react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import {
  pipeline,
  SummarizationPipeline,
  SummarizationSingle,
  AutoModelForSeq2SeqLM,
  AutoTokenizer,
} from "@huggingface/transformers";

const text =
  "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, " +
  "and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. " +
  "During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest " +
  "man-made structure in the world, a title it held for 41 years until the Chrysler Building in New " +
  "York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to " +
  "the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the " +
  "Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second " +
  "tallest free-standing structure in France after the Millau Viaduct.";

function App() {
  const [input, setInput] = useState(text);
  const [output, setOutput] = useState("");
  const [pending, setPending] = useState(false);
  const generatorRef = useRef<SummarizationPipeline | null>(null);
  const modelRef = useRef<AutoModelForSeq2SeqLM | null>(null);
  const tokenizerRef = useRef<AutoTokenizer | null>(null);

  const handleRun = async () => {
    try {
      setPending(true);
      if (!generatorRef.current) {
        console.log("create pipeline start");
        generatorRef.current = await pipeline("summarization", "Xenova/distilbart-cnn-6-6", {
          progress_callback: data => {
            switch (data.status) {
              case "initiate":
                {
                  const { name, file } = data;
                  console.log("initiate", name, file);
                }
                break;
              case "download":
                {
                  const { name, file } = data;
                  console.log("download", name, file);
                }
                break;
              case "progress":
                {
                  const { name, file, progress, loaded, total } = data;
                  console.log("progress", name, file, progress, loaded, total);
                }
                break;
              case "done":
                {
                  const { name, file } = data;
                  console.log("done", name, file);
                }
                break;
              case "ready":
                {
                  const { task, model } = data;
                  console.log("ready", task, model);
                }
                break;
            }
          },
          device: "webgpu",
          dtype: "fp32",
        });
        console.log("create pipeline done");
      }
      const start = Date.now();
      const summary = await generatorRef.current(input);
      const end = Date.now();
      console.log(`推理时间: ${end - start}ms`);
      const output = (summary[0] as unknown as SummarizationSingle).summary_text;
      setOutput(output);
    } catch (e) {
      console.error(e);
      alert(e);
    } finally {
      setPending(false);
    }
  };

  // @ts-expect-error TS6133
  const handleRunWithoutPipeline = async () => {
    try {
      setPending(true);
      if (!tokenizerRef.current || !modelRef.current) {
        console.log("create pipeline start");
        // @ts-expect-error TS7006
        const progress_callback = data => {
          switch (data.status) {
            case "initiate":
              {
                const { name, file } = data;
                console.log("initiate", name, file);
              }
              break;
            case "download":
              {
                const { name, file } = data;
                console.log("download", name, file);
              }
              break;
            case "progress":
              {
                const { name, file, progress, loaded, total } = data;
                console.log("progress", name, file, progress, loaded, total);
              }
              break;
            case "done":
              {
                const { name, file } = data;
                console.log("done", name, file);
              }
              break;
            case "ready":
              {
                const { task, model } = data;
                console.log("ready", task, model);
              }
              break;
          }
        };

        const config = {
          progress_callback,
          device: "webgpu",
          dtype: "fp32",
        };
        // @ts-expect-error TS2345
        modelRef.current = await AutoModelForSeq2SeqLM.from_pretrained("Xenova/distilbart-cnn-6-6", config);
        tokenizerRef.current = await AutoTokenizer.from_pretrained("Xenova/distilbart-cnn-6-6");
        console.log("create pipeline done");
      }
      const start = Date.now();

      // @ts-expect-error TS2349
      const inputs = await tokenizerRef.current([text], {
        truncation: true,
        return_tensors: true,
      });
      // @ts-expect-error TS2349
      const modelOutputs = await modelRef.current.generate(inputs);
      // @ts-expect-error TS2349
      const summary = await tokenizerRef.current.batch_decode(modelOutputs, {
        skip_special_tokens: true,
      });

      const end = Date.now();
      console.log(`推理时间: ${end - start}ms`);
      const output = summary[0];
      setOutput(output);
    } catch (e) {
      console.error(e);
      alert(e);
    } finally {
      setPending(false);
    }
  };

  return (
    <>
      <div className="flex flex-col items-center justify-center min-h-screen py-12 space-y-6">
        <div className="space-y-6 text-center fixed top-12 left-0 w-full flex flex-col items-center justify-center">
          <h1 className="text-4xl font-bold tracking-tight">Transformers.js</h1>
          <div className="flex items-center space-x-4">
            <div className="flex items-center gap-4">
              <div className="flex flex-col gap-2">
                <span className="text-2xl">输入</span>
                <div className="w-[400px] h-[350px] aspect-auto bg-muted rounded-lg border-2 border-dashed border-muted-foreground flex items-center justify-center">
                  <Textarea
                    className="w-full h-full"
                    value={input}
                    onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setInput(e.target.value)}
                  />
                </div>
              </div>
              <div className="flex flex-col gap-2">
                <span className="text-2xl">输出</span>
                <div className="w-[400px] h-[350px] aspect-auto bg-muted rounded-lg border-2 border-dashed border-muted-foreground flex items-center justify-center">
                  <Textarea className="w-full h-full" value={output} readOnly />
                </div>
              </div>
            </div>
          </div>
          <div className="space-y-2">
            <Button className="w-32" onClick={handleRun} disabled={pending}>
              {pending ? "推理中..." : "推理"}
            </Button>
          </div>
        </div>
      </div>
    </>
  );
}

export default App;
