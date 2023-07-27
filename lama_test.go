package llama_test

import (
	"os"

	. "github.com/go-skynet/go-llama.cpp"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("LLama binding", func() {
	Context("Declaration", func() {
		It("fails with no model", func() {
			model, err := New("not-existing")
			Expect(err).To(HaveOccurred())
			Expect(model).To(BeNil())
		})
	})
	Context("Inferencing", func() {
		It("Works with "+os.Getenv("TEST_MODEL"), func() {
			model, err := New(os.Getenv("TEST_MODEL"), EnableF16Memory, SetContext(128), SetMMap(true))
			Expect(err).ToNot(HaveOccurred())
			Expect(model).ToNot(BeNil())
			text, err := model.Predict(`Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction: How much is 2+2?

### Response: `)
			Expect(err).ToNot(HaveOccurred())
			Expect(text).To(ContainSubstring("4"))
		})
	})
})
