


declare -a model_paths=(
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--Qwen--Qwen2.5-72B-Instruct/snapshots/d3d951150c1e5848237cd6a7ad11df4836aee842"
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/e0bc86c23ce5aae1db576c8cca6f06f1f73af2db"
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--microsoft--Phi-4-mini-instruct/snapshots/5a149550068a1eb93398160d8953f5f56c3603e9"
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--microsoft--Phi-3-small-128k-instruct/snapshots/ad85cab62be398dc90203c4377a4ccbf090fbb36"
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--microsoft--Phi-3-medium-128k-instruct/snapshots/fa7d2aa4f5ea69b2e36b20d050cdae79c9bfbb3f"
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/phi_test/models--microsoft--Phi-3-mini-128k-instruct/snapshots/a90b62ae09941edff87a90ced39ba5807e6b2ade"
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7"
    # "/home/aiscuser/zhengyu_blob_home/hugging_face_models/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b"
    "/home/v-huzhengyu/zhengyu_blob_home/hugging_face_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)

declare -a model_names=(
    # "qwen2.5-7b"
    # "qwen2.5-72b-chat"
    # "qwen2.5-32b-chat"
    # "qwen2.5-14b-chat"
    # "mistral-7b-v0.3"
    # "phi-4-mini"
    # "phi-3-small"
    # "phi-3-medium" 
    # "phi-3-mini"
    # "deepseek-v2-lite-chat"
    # "llama-3.1-70b"
    "llama-3.1-8b"
)


for i in "${!model_paths[@]}"; do
    model_path="${model_paths[$i]}"
    model_name="${model_names[$i]}"
    
    echo "Running evaluation for model: $model_name"
    

    python host_and_eval.py \
    --model-path "$model_path" \
    --model-name "$model_name" \
    --gpu 1 \
    --port 8010 \
    --just_host=True
done


# bash /home/aiscuser/zhengyu_blob_home/kkk_vllm_first_4.sh