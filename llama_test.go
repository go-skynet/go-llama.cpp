package llama_test

import (
	"fmt"
	"os"

	. "github.com/go-skynet/go-llama.cpp"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("LLama binding", func() {
	testModelPath := os.Getenv("TEST_MODEL")

	Context("Declaration", func() {
		It("fails with no model", func() {
			model, err := New("not-existing")
			Expect(err).To(HaveOccurred())
			Expect(model).To(BeNil())
		})
	})
	if testModelPath != "" {
		Context("Inferencing tests (using "+testModelPath+") ", func() {

			getModel := func() (*LLama, error) {
				model, err := New(
					testModelPath,
					EnableF16Memory,
					SetContext(128),
					SetMMap(true),
					SetNBatch(512),
				)
				Expect(err).ToNot(HaveOccurred())
				Expect(model).ToNot(BeNil())
				return model, err
			}

			It("predicts successfully", func() {
				model, err := getModel()
				text, err := model.Predict(`Below is an instruction that describes a task. Write a response that appropriately completes the request.

	### Instruction: How much is 2+2?

	### Response: `, SetRopeFreqBase(10000.0), SetRopeFreqScale(1))
				Expect(err).ToNot(HaveOccurred())
				Expect(text).To(ContainSubstring("4"))
			})
			It("tokenizes strings successfully", func() {
				model, err := getModel()
				l, tokens, err := model.TokenizeString("A STRANGE GAME.\nTHE ONLY WINNING MOVE IS NOT TO PLAY.\n\nHOW ABOUT A NICE GAME OF CHESS?",
					SetRopeFreqBase(10000.0), SetRopeFreqScale(1))

				Expect(err).ToNot(HaveOccurred())
				Expect(l).To(BeNumerically(">", 0))
				fmt.Printf("Temp Token Dump: %+v", tokens)
				Expect(l).To(Equal(len(tokens)))
			})
		})
	}
})
