package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"

	common "github.com/go-skynet/go-common"
	llama "github.com/go-skynet/go-llama.cpp"
)

var (
	threads   = 4
	tokens    = 128
	gpulayers = 0
)

func main() {
	var model string

	flags := flag.NewFlagSet(os.Args[0], flag.ExitOnError)
	flags.StringVar(&model, "m", "./models/7B/ggml-model-q4_0.bin", "path to q4_0.bin model file to load")
	flags.IntVar(&gpulayers, "ngl", 0, "Number of GPU layers to use")
	flags.IntVar(&threads, "t", runtime.NumCPU(), "number of threads to use during computation")
	flags.IntVar(&tokens, "n", 512, "number of tokens to predict")

	err := flags.Parse(os.Args[1:])
	if err != nil {
		fmt.Printf("Parsing program arguments failed: %s", err)
		os.Exit(1)
	}
	l, err := llama.LLamaBackendInitializer.New(model, common.EnableF16Memory, common.SetContext(128), common.EnableEmbeddings, common.SetGPULayers(gpulayers))
	if err != nil {
		fmt.Println("Loading the model failed:", err.Error())
		os.Exit(1)
	}
	fmt.Printf("Model loaded successfully.\n")

	reader := bufio.NewReader(os.Stdin)

	for {
		text := readMultiLineInput(reader)

		_, err := l.Predict(text, common.Debug, common.SetTokenCallback(func(token string) bool {
			fmt.Print(token)
			return true
		}), common.SetTokens(tokens), common.SetThreads(threads), common.SetTopK(90), common.SetTopP(0.86), common.SetStopWords("llama"))
		if err != nil {
			panic(err)
		}
		embeds, err := l.StringEmbeddings(text)
		if err != nil {
			fmt.Printf("Embeddings: error %s \n", err.Error())
		}
		fmt.Printf("Embeddings: %v", embeds)
		fmt.Printf("\n\n")
	}
}

// readMultiLineInput reads input until an empty line is entered.
func readMultiLineInput(reader *bufio.Reader) string {
	var lines []string
	fmt.Print(">>> ")

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				os.Exit(0)
			}
			fmt.Printf("Reading the prompt failed: %s", err)
			os.Exit(1)
		}

		if len(strings.TrimSpace(line)) == 0 {
			break
		}

		lines = append(lines, line)
	}

	text := strings.Join(lines, "")
	fmt.Println("Sending", text)
	return text
}
