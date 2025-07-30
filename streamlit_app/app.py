import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle

class NCF(nn.Module):
    def __init__(self, num_users, num_items, num_locations, embedding_dim=64):
        super(NCF, self).__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        self.item_embed = nn.Embedding(num_items, embedding_dim)
        self.loc_embed = nn.Embedding(num_locations, 8)
        self.age_fc = nn.Linear(1, 4)
        self.final_fc = nn.Linear(1 + 8 + 4, 1)  # sim_score + loc_vec + age_vec

    def forward(self, user, item, location, age):
        user_vec = self.user_embed(user)
        item_vec = self.item_embed(item)
        sim_score = (user_vec * item_vec).sum(dim=1, keepdim=True)
        loc_vec = self.loc_embed(location)
        age_vec = self.age_fc(age.unsqueeze(1))
        x = torch.cat([sim_score, loc_vec, age_vec], dim=1)
        out = self.final_fc(x)
        return out.squeeze()

@st.cache_resource
def load_all():
    with open("/media/kursat/TOSHIBA EXT61/projects/ice_global/notebook/user_encoder.pkl", "rb") as f:
        user_encoder = pickle.load(f)
    with open("/media/kursat/TOSHIBA EXT61/projects/ice_global/notebook/book_encoder.pkl", "rb") as f:
        book_encoder = pickle.load(f)
    num_users = len(user_encoder.classes_)
    num_books = len(book_encoder.classes_)

    books_df = pd.read_csv("/media/kursat/TOSHIBA EXT61/projects/ice_global/data/books_df_for_dashboard.csv", encoding="latin-1")
    users_df = pd.read_csv("/media/kursat/TOSHIBA EXT61/projects/ice_global/data/users_df_for_dashboard.csv", encoding="latin-1")

    books_df["ISBN"] = books_df["ISBN"].astype(str).str.strip()
    users_df["User-ID"] = users_df["User-ID"].astype(str).str.strip()

    user_encoder_classes = [str(x).strip() for x in user_encoder.classes_]
    book_encoder_classes = [str(x).strip() for x in book_encoder.classes_]

    users_df["location_index"] = users_df["Location"].astype("category").cat.codes
    users_df["Age"] = pd.to_numeric(users_df["Age"], errors='coerce')
    users_df.loc[(users_df["Age"] < 5) | (users_df["Age"] > 100), "Age"] = np.nan
    users_df["Age"].fillna(users_df["Age"].median(), inplace=True)
    users_df["Age_norm"] = (users_df["Age"] - users_df["Age"].min()) / (users_df["Age"].max() - users_df["Age"].min())

    num_locations = users_df["location_index"].nunique()

    model = NCF(num_users, num_books, num_locations, embedding_dim=64)
    model.load_state_dict(torch.load("/media/kursat/TOSHIBA EXT61/projects/ice_global/notebook/model.pth", map_location="cpu"))
    model.eval()
    return model, user_encoder, book_encoder, books_df, users_df, user_encoder_classes, book_encoder_classes

model, user_encoder, book_encoder, books_df, users_df, user_encoder_classes, book_encoder_classes = load_all()

def recommend_books_ncf(user_id, top_k=10):
    user_id = str(user_id).strip()
    if user_id not in user_encoder_classes:
        return []
    user_idx = user_encoder.transform([user_id])[0]
    user_row = users_df[users_df["User-ID"] == user_id]
    if user_row.empty:
        return []
    location_idx = user_row["location_index"].values[0]
    age_norm = user_row["Age_norm"].values[0]

    read_books = set() 
    unread_books = [isbn for isbn in book_encoder_classes if isbn not in read_books]
    if not unread_books:
        return []
    unread_book_indices = book_encoder.transform(unread_books)
    user_tensor = torch.tensor([user_idx] * len(unread_book_indices), dtype=torch.long)
    book_tensor = torch.tensor(unread_book_indices, dtype=torch.long)
    loc_tensor = torch.tensor([location_idx] * len(unread_book_indices), dtype=torch.long)
    age_tensor = torch.tensor([age_norm] * len(unread_book_indices), dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        scores = model(user_tensor, book_tensor, loc_tensor, age_tensor).cpu().numpy()
    top_indices = scores.argsort()[::-1][:top_k]
    recommended_isbns = [unread_books[i] for i in top_indices]
    return recommended_isbns

st.title("📚 Book Recommender Dashboard")

tab1, tab2 = st.tabs(["Recommend Book to the User", "Similar Books to a Book"])

with tab1:
    st.header("Personalized Book Recommendations")
    user_id = st.selectbox(
        "User ID seç",
        options=user_encoder_classes
    )
    top_k = st.slider("How many recommendations?", min_value=1, max_value=20, value=10)
    if st.button("Recommend Books"):
        rec_isbns = recommend_books_ncf(user_id, top_k=top_k)
        st.subheader("Recommended Books:")
        for isbn in rec_isbns:
            isbn_str = str(isbn).strip()
            book_row = books_df[books_df["ISBN"] == isbn_str]
            if not book_row.empty:
                title = book_row["Book-Title"].values[0]
                author = book_row["Book-Author"].values[0]
                publisher = book_row["Publisher"].values[0]
                img_url = book_row["Image-URL-M"].values[0] if "Image-URL-M" in book_row.columns else None
                with st.container():
                    cols = st.columns([1,4])
                    if img_url:
                        cols[0].image(img_url, width=80)
                    cols[1].markdown(f"**{title}**\n\n*by {author}*\n\n_Publisher: {publisher}_\n\n_ISBN: {isbn_str}_")

with tab2:
    st.header("Similar Books to the Book (in the Embedding Space)")
    item_isbn = st.selectbox("ISBN seç", options=book_encoder_classes)
    top_k_sim = st.slider("How many books are similar?", min_value=1, max_value=20, value=10, key="sim")
    if st.button("Show Similar Books"):
        with torch.no_grad():
            item_idx = book_encoder.transform([str(item_isbn).strip()])[0]
            item_emb = model.item_embed(torch.tensor([item_idx]))
            all_emb = model.item_embed.weight.data
            sims = torch.matmul(all_emb, item_emb.squeeze())
            topk = torch.topk(sims, top_k_sim+1).indices.cpu().numpy()
            topk = [idx for idx in topk if idx != item_idx][:top_k_sim]
            sim_isbns = book_encoder.inverse_transform(topk)
        st.subheader("Similar Books:")
        for isbn in sim_isbns:
            isbn_str = str(isbn).strip()
            book_row = books_df[books_df["ISBN"] == isbn_str]
            if not book_row.empty:
                title = book_row["Book-Title"].values[0]
                author = book_row["Book-Author"].values[0]
                publisher = book_row["Publisher"].values[0]
                img_url = book_row["Image-URL-M"].values[0] if "Image-URL-M" in book_row.columns else None
                with st.container():
                    cols = st.columns([1,4])
                    if img_url:
                        cols[0].image(img_url, width=80)
                    cols[1].markdown(f"**{title}**\n\n*by {author}*\n\n_Publisher: {publisher}_\n\n_ISBN: {isbn_str}_")
