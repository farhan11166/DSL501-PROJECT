# DSL501-PROJECT

Dataset: We have used [lmsys-chat-1m dataset] (https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
As the data set has 1 million entries we have only shown the first 500 entries of the dataset .The whole dataset can be retrieved in following method.
## 4. Accessing Gated Datasets (Hugging Face CLI Login)

The `lmsys-chat-1m` dataset requires authentication before it can be downloaded or accessed programmatically. You must log in using the Hugging Face Command Line Interface (CLI).

### A. Terminal Login

1.  **Open your terminal** (e.g., Kitty).
2.  **Run the login command:**
    ```bash
    huggingface-cli login
    ```
3.  The command will prompt you to enter your **Hugging Face access token**.

### B. How to Generate an Access Token

If you do not already have a token, you must generate one in your web browser:

1.  Go to the **Hugging Face website** and **log into your account**.
2.  Navigate to your **Settings** (usually accessible by clicking your profile picture).
3.  Click on **Access Tokens** in the sidebar.
4.  Click the **+ New token** button.
5.  Provide a name and select the necessary role (e.g., 'Read' or 'Write/Read').
6.  **Copy the generated token string** and paste it into your terminal when prompted by the `huggingface-cli login` command.
7.  Run the DatasetRetrieval.ipynb it will give the data in train and test split seperately .
