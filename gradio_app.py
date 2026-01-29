import gradio as gr
import pickle
from sentiment_analysis import SentimentAnalyzer

def load_model():
    """Model və analizeri yükləyir"""
    analyzer = SentimentAnalyzer()
    analyzer.load_data(
        train_path='dataset/training.csv',
        val_path='dataset/validation.csv',
        test_path='dataset/test.csv'
    )
    analyzer.prepare_data()
    analyzer.train_models()
    return analyzer

# Global analyzer
analyzer = load_model()

def predict_sentiment(text):
    if not text.strip():
        return "Please enter text!", {}
    
    emotion, probabilities = analyzer.predict_emotion(text)
    
    # Nəticəni formatlaşdıraq
    result = f"🎯 **Predicted Emotion:** {emotion.upper()}\n\n"
    result += "📊 **Probability Distribution:**\n"
    
    # Ehtimalları sıralayaq
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    for emo, prob in sorted_probs:
        bar = "█" * int(prob * 50)
        result += f"- {emo:12s}: {prob:.2%} {bar}\n"
    
    return result, probabilities

# Nümunə mətnlər
examples = [
    ["I am so happy and excited about my new job!"],
    ["I feel very sad and lonely without you"],
    ["I love spending time with my family"],
    ["This makes me so angry and frustrated"],
    ["I am scared and worried about the future"],
    ["Wow, I cannot believe this just happened!"]
]

# Gradio interfeys
with gr.Blocks(theme=gr.themes.Soft(), title="🎭 Sentiment Analysis") as demo:
    
    gr.Markdown("""
    # 🎭 Sentiment Analysis
    ### Text Analysis by 6 Emotion Categories
    
    This model determines what emotions are present in the text written by the user:
    - 😢 **Sadness** (Kədər)
    - 😊 **Joy** (Sevinc)  
    - ❤️ **Love** (Sevgi)
    - 😠 **Anger** (Qəzəb)
    - 😨 **Fear** (Qorxu)
    - 😲 **Surprise** (Təəccüb)
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="📝 Enter your text.",
                placeholder="For example: I am feeling great today!",
                lines=5
            )
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                submit_btn = gr.Button("🚀 Analyze", variant="primary")
        
        with gr.Column(scale=2):
            output_text = gr.Textbox(
                label="🎯 Result",
                lines=10
            )
            output_plot = gr.Label(
                label="📊 Probability Graph",
                num_top_classes=6
            )
    
    gr.Markdown("### 💡 Examples of Sample Texts:")
    gr.Examples(
        examples=examples,
        inputs=text_input,
        label="Try Sample Texts"
    )
    
    submit_btn.click(
        fn=predict_sentiment,
        inputs=text_input,
        outputs=[output_text, output_plot]
    )
    
    clear_btn.click(
        fn=lambda: ("", {}, ""),
        outputs=[text_input, output_plot, output_text]
    )
    
    gr.Markdown("""
    ---
    ###  About:
    This project is built using a **Linear SVM** model and trained on **16,000** training texts.
    It achieves **88.35%** accuracy on the test set.
    
    **Technologies:** Python, scikit-learn, TF-IDF, Gradio
    """)

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )