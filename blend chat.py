from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

class BlenderBotChatbot:
    def __init__(self, model_name="facebook/blenderbot-400M-distill"):
        """
        Initialize the chatbot with the specified model.
        Args:
            model_name (str): Hugging Face model ID or local path.
        """
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            print("✅ Model and tokenizer loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def respond(self, user_input: str) -> str:
        """
        Generate a response to user input.
        Args:
            user_input (str): User's message.
        Returns:
            str: Bot's response.
        """
        try:
            inputs = self.tokenizer(
                [user_input],
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,  # Control response length
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            return f"⚠️ Error generating response: {e}"

def main():
    # Initialize chatbot
    bot = BlenderBotChatbot()  # Uses 'facebook/blenderbot-400M-distill' by default
    
    # Interactive chat loop
    print("\nChat with BlenderBot! Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        
        response = bot.respond(user_input)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    main()
