"""
General Knowledge Chatbot using Pre-trained Language Model
Uses Hugging Face transformers library with a text-generation model
"""

from transformers import pipeline
import re


class GeneralChatbot:
    """A general knowledge chatbot using pre-trained language models."""
    
    def __init__(self, model_name="gpt2"):
        """
        Initialize the general chatbot with a pre-trained model.
        
        Args:
            model_name: The Hugging Face model to use for text generation
        """
        print(f"Loading model: {model_name}...")
        self.text_gen_pipeline = pipeline("text-generation", model=model_name, max_length=150)
        self.conversation_history = []
        print("Model loaded successfully!")
    
    def start_conversation(self):
        """
        Start a new conversation session.
        """
        self.conversation_history = []
        print("Conversation started. I can answer general knowledge questions!\n")
    
    def answer_question(self, question):
        """
        Answer a general knowledge question.
        
        Args:
            question: The question to answer
            
        Returns:
            The generated answer
        """
        try:
            # Create prompt for the model
            prompt = f"Q: {question}\nA:"
            
            # Generate response
            result = self.text_gen_pipeline(prompt, num_return_sequences=1, do_sample=True, temperature=0.8)
            response = result[0]['generated_text']
            
            # Extract the answer part
            if "A:" in response:
                answer = response.split("A:")[-1].strip()
            else:
                answer = response.strip()
            
            # Store in history
            self.conversation_history.append({"question": question, "answer": answer})
            
            return answer
        except Exception as e:
            return f"I couldn't generate a response. Error: {str(e)}"
    
    def chat(self):
        """Start an interactive chat session."""
        print("=" * 60)
        print("General Knowledge Chatbot Started!")
        print("Type 'exit' to quit, 'history' to see conversation history")
        print("=" * 60 + "\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'exit':
                print("\nThank you for using the General Knowledge Chatbot. Goodbye!")
                break
            
            if user_input.lower() == 'history':
                if self.conversation_history:
                    print("\n--- Conversation History ---")
                    for i, item in enumerate(self.conversation_history, 1):
                        print(f"{i}. Q: {item['question']}\n   A: {item['answer']}\n")
                else:
                    print("No conversation history yet.\n")
                continue
            
            if not user_input:
                continue
            
            print(f"Bot: {self.answer_question(user_input)}\n")


def main():
    """Main function to demonstrate the general knowledge chatbot."""
    
    # Initialize chatbot
    chatbot = GeneralChatbot()
    chatbot.start_conversation()
    
    print("General Knowledge Chatbot Demo")
    print("-" * 60)
    
    # Example general knowledge questions
    example_questions = [
        "Where is Taj Mahal?",
        "What is DFS and BFS techniques?",
        "How tall is Mount Everest?",
        "Who was the first President of USA?"
    ]
    
    print("\nExample Questions and Answers:")
    print("=" * 60)
    
    for question in example_questions:
        print(f"\nYou: {question}")
        print(f"Bot: {chatbot.answer_question(question)}")
    
    print("\n" + "=" * 60)
    print("\nWould you like to start an interactive session? (y/n): ", end="")
    response = input().strip().lower()
    
    if response == 'y':
        chatbot.chat()
    else:
        print("Thank you for using the General Knowledge Chatbot!")


if __name__ == "__main__":
    main()
