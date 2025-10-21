import pandas as pd

# 파일 경로와 분할 크기 설정
input_file = "./source/1m_history.csv"  # 읽을 대용량 CSV 파일 경로
output_prefix = "chunk_"       # 분할 파일의 접두사
chunk_size = 100000            # 한 번에 처리할 행 수 (예: 10만 행)

# CSV 파일을 분할하여 저장
try:
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        output_file = f"./source/chunk_{i}.csv"
        chunk.to_csv(output_file, index=False)  # 인덱스를 제외하고 저장
        print(f"Saved {output_file}")
except FileNotFoundError:
    print(f"File {input_file} not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")
