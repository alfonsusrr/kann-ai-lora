echo "Installing Dependencies"
pip install python-dotenv nltk rouge_score  xformers trl peft accelerate bitsandbytes triton "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" datasets transformers pinecone sentence-transformers flash_attn

echo "Initializing Git-LFS"
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs
git-lfs install

echo "Initializing Git"
git config --global user.name "Alfonsus Rendy"
git config --global user.email "alfonsus737@gmail.com"

