# app.py
import streamlit as st
import ui                   # UIモジュール
import llm                  # LLMモジュール
import database             # データベースモジュール
import metrics              # 評価指標モジュール
import data                 # データモジュール
import torch
from transformers import pipeline
from config import MODEL_NAME
from huggingface_hub import HfFolder
from PIL import Image
import tempfile

# --- アプリケーション設定 ---
st.set_page_config(page_title="InternVL Chatbot", layout="wide")

# --- 初期化処理 ---
# NLTKデータのダウンロード（初回起動時など）
metrics.initialize_nltk()

# データベースの初期化（テーブルが存在しない場合、作成）
database.init_db()

# データベースが空ならサンプルデータを投入
data.ensure_initial_data()

# LLMモデルのロード（キャッシュを利用）
# モデルをキャッシュして再利用
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}") # 使用デバイスを表示
        pipe = pipeline(
            "image-text-to-text",
            model=MODEL_NAME,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device
        )
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None
pipe = llm.load_model()

# --- Streamlit アプリケーション ---
st.title("🧠 InternVL Chatbot with Image Input")
st.write("InternVLモデルを使用したマルチモーダルチャットボットです。画像とテキストを組み合わせて質問できます。")
st.markdown("---")

# --- サイドバー ---
st.sidebar.title("ナビゲーション")
# セッション状態を使用して選択ページを保持
if 'page' not in st.session_state:
    st.session_state.page = "チャット" # デフォルトページ

page = st.sidebar.radio(
    "ページ選択",
    ["チャット", "履歴閲覧", "サンプルデータ管理"],
    key="page_selector",
    index=["チャット", "履歴閲覧", "サンプルデータ管理"].index(st.session_state.page), # 現在のページを選択状態にする
    on_change=lambda: setattr(st.session_state, 'page', st.session_state.page_selector) # 選択変更時に状態を更新
)


# --- メインコンテンツ ---
if st.session_state.page == "チャット":
    if pipe:
        uploaded_image = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg"])
        text_input = st.text_area("質問を入力", "")

        if st.button("送信"):
            if not text_input:
                st.warning("質問を入力してください。")
            else:
                messages = [{"role": "user", "content": []}]

                if uploaded_image:
                    image = Image.open(uploaded_image).convert("RGB")
                    st.image(image, caption="アップロードされた画像")  # ✅ 表示を外に出す

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        image.save(tmp.name)
                        image_path = tmp.name

                    messages[0]["content"].append({
                        "type": "image",
                        "image": image_path  # str型のファイルパス
                    })

                messages[0]["content"].append({
                    "type": "text",
                    "text": text_input
                })
                print(messages)

                try:
                    output = pipe(messages, max_new_tokens=256, return_full_text=False)
                    result = output[0]["generated_text"].strip()
                    st.markdown("### モデルの応答")
                    st.write(result if result else "モデルからの応答がありませんでした。")
                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")
    else:
        st.error("チャット機能を利用できません。モデルの読み込みに失敗しました。")
elif st.session_state.page == "履歴閲覧":
    ui.display_history_page()
elif st.session_state.page == "サンプルデータ管理":
    ui.display_data_page()

# --- フッターなど（任意） ---
st.sidebar.markdown("---")
st.sidebar.info("開発者: [Your Name]")