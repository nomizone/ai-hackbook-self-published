FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ダミーの機密ファイルを配置（RCEで読み取られるデモ用）
RUN mkdir -p /app/data/secret && \
    echo '顧客ID,氏名,メールアドレス,電話番号,住所' > /app/data/secret/顧客情報.csv && \
    echo '1001,山田太郎,yamada@example.com,090-1234-5678,東京都世田谷区北沢2-14-7' >> /app/data/secret/顧客情報.csv && \
    echo '1002,佐藤花子,sato@example.com,080-2345-6789,神奈川県横浜市中区本町3-28-1' >> /app/data/secret/顧客情報.csv && \
    echo '1003,鈴木一郎,suzuki@example.com,070-3456-7890,大阪府大阪市北区梅田1-5-12' >> /app/data/secret/顧客情報.csv && \
    echo '=== 売上レポート（社外秘） ===' > /app/data/secret/売上レポート.txt && \
    echo '2026年Q1 売上: 4億2,300万円' >> /app/data/secret/売上レポート.txt && \
    echo '利益率: 18.5%' >> /app/data/secret/売上レポート.txt && \
    echo '新規顧客獲得数: 342社' >> /app/data/secret/売上レポート.txt && \
    echo 'トップセールス: 営業3課 田中（1億800万円）' >> /app/data/secret/売上レポート.txt && \
    echo 'FLAG{eval_is_evil_rce_achieved}' > /root/root.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
