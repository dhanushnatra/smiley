
# Step1 :
- go to <a href="https://www.kaggle.com/datasets/fatihkgg/affectnet-yolo-format">Dataset Link</a>
and download dataset
- extract to folder named dataset in working directory and then
- create a virtual environment and activate it  (optional)
- Install requirements
    ```bash
    pip install -r requirements.txt
    ```

# Step 2 (Optional): 

- ## Run

    ```bash
    python3 create_dataset.py
    ```

# Step 3 (Optional):

- Run train_model.ipynb Notebook

# Step 4 :

 - ## Run :
    -   run this to view it in your web browser

    ```
    uvicorn app:app 
    ```
    - open the link http://127.0.0.1:8000 
