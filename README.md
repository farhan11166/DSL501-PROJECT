# DSL501-PROJECT

## Dataset

We use the [lmsys-chat-1m dataset](https://huggingface.co/datasets/lmsys/lmsys-chat-1m).  
Due to its large size (1 million entries), we showcase only the first 500 entries.  
You can retrieve the full dataset as described below.

---

## Accessing Gated Datasets (Hugging Face CLI Login)

The `lmsys-chat-1m` dataset requires authentication for download or programmatic access.  
Follow these steps to log in using the Hugging Face Command Line Interface (CLI):

### A. Terminal Login

1. **Open your terminal** .
2. **Run the login command:**
    ```bash
    huggingface-cli login
    ```
3. Enter your **Hugging Face access token** when prompted.

### B. Generating an Access Token

If you do not have a token:

1. Go to the [Hugging Face website](https://huggingface.co/) and log into your account.
2. Navigate to **Settings** (click your profile picture).
3. Click **Access Tokens** in the sidebar.
4. Click **+ New token**.
5. Provide a name and select the required role (e.g., 'Read' or 'Write/Read').
6. **Copy the generated token** and paste it into your terminal when prompted by `huggingface-cli login`.

---

After logging in, run `DatasetRetrieval.ipynb` to obtain the data in separate train and test splits.

---

> **Note:**  
> This README will be updated as the project progresses.  
> For detailed information about the project, please refer to [SoP_ML.pdf](https://github.com/farhan11166/DSL501-PROJECT/blob/main/SoP_ML.pdf).