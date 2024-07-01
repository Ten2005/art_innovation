from flask import Flask, render_template, request
import openai
from google.oauth2 import service_account
from sentence_transformers import SentenceTransformer, util
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

app = Flask(__name__)

google_credentials = service_account.Credentials.from_service_account_file("/Users/suzukten/notional-life-422509-d1-0e8a1099fa74.json")
openai.api_key = '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'

# データベースの作成
Base = declarative_base()
class History(Base):
    __tablename__ = "histories"
    id = Column(Integer, primary_key=True)
    history_absurd = Column(String)

    def __repr__(self):
        return f"<History(id={self.id}, history_absurd={self.history_absurd})>"

engine = create_engine("sqlite:///app.db")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def generate_prompt(conversation_history, user_input, conversation_history_sum):
    """ユーザー入力と会話履歴に基づいてプロンプトを生成する"""
    input_his = conversation_history
    his_summary = conversation_history_sum
    # {history_text_for_add}
    return f"会話履歴:{input_his}：会話の要約: {his_summary}\n質問に対して短く答えて下さい: {user_input}。"

def generate_response(user_input, conversation_history, conversation_history_sum):
    """OpenAIのChatGPTを使用して応答を生成する"""
    system_message = "あなたは本音と本質を見抜くプロです、以下の三つを必ず行なってください:1.短く適切に共感を示す 2.悩み事の真の原因について心理的,環境的な観点から一つ質問する 3.具体的な原因を予想して一つだけ曖昧に述べる"
    prompt = generate_prompt(conversation_history, user_input, conversation_history_sum)
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

def generate_response_goal(user_input, conversation_history, conversation_history_sum):
    """OpenAIのChatGPTを使用して応答を生成する"""
    system_message = "あなたは本音と本質を見抜くプロです、必ず以下の2つを行なってください:1.短く適切に共感を示す 2.なりたい理想の姿を聞き出す"
    prompt = generate_prompt(conversation_history, user_input, conversation_history_sum)
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

def sum_prompt(text):
    """ユーザー入力と会話履歴に基づいてプロンプトを生成する"""
    return f"以下の文章をユーザーに関して70字以内で要約してください: {text}。"

def summary(text):
    """OpenAIのChatGPTを用いて要約する"""
    prompt = sum_prompt(text)
    sum = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return sum['choices'][0]['message']['content']


def generate_text_adjust(res):
    """GPTの出力に基づいてプロンプトを生成する"""
    return f"下記の文章に対し以下の2つの変更を加えてください。変更:1.短くする 2.自然な日本語にする。文章: {res}。"



def adjust_response(res):
    """OpenAIのChatGPTを使用して応答を調節する"""
    prompt = generate_text_adjust(res)
    
    response_2 = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response_2['choices'][0]['message']['content']



def sim(a, b):
    #Compute embedding for both lists
    embedding_1= model.encode(a, convert_to_tensor=True)
    embedding_2 = model.encode(b, convert_to_tensor=True)
    return util.pytorch_cos_sim(embedding_1, embedding_2)

# 情報の整理
def org_prompt(text):
    return f"以下の会話履歴に対し、次の二つを行なってください。１ ユーザーの現状について詳細に整理し、箇条書きでまとめる。２ ユーザーのありたい理想、目標について箇条書きで整理する。会話履歴: {text}。"

def organize(text):
    """OpenAIのChatGPTを用いて現状と目標について整理する"""
    prompt = org_prompt(text)
    sum = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return sum['choices'][0]['message']['content']

def solution_prompt(text):
    return f"下記のユーザーに対してすぐに実践可能な解決策を一つ提示し、簡潔に述べてください。ユーザー: {text}。"

def solution(text):
    """OpenAIのChatGPTを用いて現状と目標について解決策を出力"""
    prompt = solution_prompt(text)
    sum = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return sum['choices'][0]['message']['content']

@app.route('/')
def index():
    global conversation_history, conversation_history_absurd
    conversation_history = []
    conversation_history_absurd = []
    return render_template('index.html')

conversation_history = []

@app.route('/process', methods=['POST'])
def process():
    # 入力を受け取る
    user_input = request.form['user_input']
    # 会話の終了
    if "0" in user_input:
        con_his = "".join(conversation_history)
        final_sum = organize(con_his)
        sol = solution(final_sum)

        history_ab = History(history_absurd=":".join([con_his,final_sum,sol]))
        session.add(history_ab)
        session.commit()
        return render_template(
            'end.html'
            )
    else:
        # ユーザーの入力を履歴に追加
        conversation_history.append(f"ユーザー:{user_input}")
        # 要約
        conversation_history_sum = summary(conversation_history[-10:])
        # 応答生成
        response = generate_response(user_input, conversation_history, conversation_history_sum)
        response_2 = adjust_response(response)
        # 類似度の算出と応答の吟味
        if len(conversation_history)>=2:
            similarity = sim(conversation_history[-2], response_2)
            print(f"算出された類似度:{similarity}")

            # 応答の変更
            if similarity>=0.7 or similarity < 0.5:
                print(f"閾値外の応答:{response_2}")
                # print("応答を生成し直します")
                # プロンプトを変更する
                response = generate_response_goal(user_input, conversation_history, conversation_history_sum)
                response_2 = adjust_response(response)
                similarity = sim(conversation_history[-2], response_2)
                print(f"算出された類似度:{similarity}")
        # 応答を履歴に追加
        conversation_history.append(f"AI:{response_2}")
        return render_template(
            'result.html', user_input=user_input, response_2=response_2, conversation_history=conversation_history
            )
        
fine_tuned_model_id = 'ft:gpt-3.5-turbo-0125:personal::9YtAoi8N'

conversation_history_absurd = []

@app.route('/absurd', methods=['POST'])
def absurd():
    user_input = request.form['user_input']
    if user_input.lower() == '0':
        history_ab = History(history_absurd="".join(conversation_history_absurd))
        session.add(history_ab)
        session.commit()
        return render_template(
            'end.html'
            )
    else:     
        conversation_history_absurd.append(f"あなた:{user_input}")
            # ファインチューニングされたモデルを使用して応答を生成
        response = openai.ChatCompletion.create(
            model=fine_tuned_model_id,
            messages=[
                {"role": "system", "content": "あなたは、不条理な会話シナリオを作成する作家です。ユーザの入力に対して口調を合わせてください。不条理な返答を返してください。"},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7  # 温度を設定して出力のランダム性を制御
            )
        ai_response = response['choices'][0]['message']['content']
        conversation_history_absurd.append(f"AI:{ai_response}")
        return render_template(
            'absurd.html', conversation_history_absurd=conversation_history_absurd
            )
    
@app.route('/pass', methods=['POST'])
def varify():
    user_input = request.form['user_input']
    if user_input == 'artinnovation':
        data_ab = session.query(History).all()
        return render_template(
            'admin.html', data_ab=data_ab
        )
    else:
        global conversation_history, conversation_history_absurd
        conversation_history = []
        conversation_history_absurd = []
        return render_template(
            'index.html'
        )

@app.route('/return_index', methods=['POST'])
def re_index():
    global conversation_history, conversation_history_absurd
    conversation_history = []
    conversation_history_absurd = []
    return render_template('index.html')
@app.route('/delete', methods=['POST'])
def delete():
    num_text = request.form['numbers']
    numbers = num_text.split('/')
    for i in range(len(numbers)):
        session.query(History).filter(History.id == numbers[i]).delete()
        session.commit()
    data_ab = session.query(History).all()
    return render_template('admin.html', data_ab=data_ab)  

if __name__ == '__main__':
    app.run(debug=True)