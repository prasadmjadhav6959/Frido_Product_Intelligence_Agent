import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
import random

# Set page config
st.set_page_config(
    page_title="Frido Product Intelligence Agent",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .query-response {
        padding: 15px;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 20px;
        border-left: 4px solid #4e73df;
    }
    .product-highlight {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .similar-product {
        padding: 12px;
        border: 1px solid #ddd;
        border-radius: 6px;
        margin: 8px 0;
    }
    .price-drop-alert {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin-bottom: 15px;
    }
    .notification-box {
        background-color: #ffe5e5;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin-bottom: 20px;
        font-size: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .notification-box button {
        background-color: #dc3545;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 5px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("frido_arcatron_price_history.csv")
        df['snapshot_time'] = pd.to_datetime(df['snapshot_time'])
        df['scraped_at'] = pd.to_datetime(df['scraped_at'])
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'frido_arcatron_price_history.csv' exists.")
        return pd.DataFrame()

def build_semantic_search(df):
    if df.empty:
        return None, None
    df['search_text'] = df['name'] + " " + df['description'] + " " + df['category']
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['search_text'])
    return vectorizer, tfidf_matrix

def semantic_search(query, vectorizer, tfidf_matrix, df, top_n=5):
    if vectorizer is None or df.empty:
        return pd.DataFrame()
    query_vec = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    related_indices = np.argsort(cosine_sim)[-top_n:][::-1]
    return df.iloc[related_indices]

def get_product_history(product_id, df):
    return df[df['product_id'] == product_id].sort_values('snapshot_time')

def get_price_drop_alerts(df, min_drop=10):
    latest = df.sort_values(by=["product_id", "snapshot_time"]).groupby("product_id").agg(
        name=('name', 'first'),
        old_price=('price', 'first'),
        new_price=('price', 'last'),
        snapshot_time=('snapshot_time', 'last'),
        category=('category', 'first'),
        stock=('availability', 'first')
    ).reset_index()  # Simply reset the index to bring product_id back as a column
    latest["drop_pct"] = ((latest["old_price"] - latest["new_price"]) / latest["old_price"]) * 100
    return latest[latest["drop_pct"] >= min_drop]

def generate_notification(product, user_name="Customer"):
    try:
        # Ensure product is a pandas Series and access attributes correctly
        name = product['name'] if isinstance(product, pd.Series) else product.name
        drop_pct = product['drop_pct'] if isinstance(product, pd.Series) else product.drop_pct
        category = product['category'] if isinstance(product, pd.Series) else product.category
        old_price = product['old_price'] if isinstance(product, pd.Series) else product.old_price
        new_price = product['new_price'] if isinstance(product, pd.Series) else product.new_price
        product_id = product['product_id'] if isinstance(product, pd.Series) else product.product_id

        notification_styles = [
            # Classic Urgency
            f"⚡️ Flash Sale Ends Tonight! Get {drop_pct:.0f}% OFF on {name}! Shop now before it's gone!",
            # Scarcity & Personal Touch
            f"⏳ Last Chance, {user_name}! Your favorite {category} item, {name}, is selling out! Grab it now with {drop_pct:.0f}% off while supplies last!",
            # Benefit-Driven & Time-Sensitive
            f"🎉 Deal Alert! Upgrade your mobility with {name} – now {drop_pct:.0f}% OFF for the next 3 hours only!",
            # Direct & Exclusive
            f"🎁 Exclusive Offer Just For You, {user_name}! Get ₹{old_price - new_price:,.0f} OFF on {name}. Valid for 24 hours! Use code: SPECIAL{str(product_id)[-4:]}",
            # Running Out Alarm
            f"🔥 Hurry! Stock is LOW on {name}! Don't let it disappear – secure it now with {drop_pct:.0f}% OFF!"
        ]
        return random.choice(notification_styles)
    except (KeyError, AttributeError) as e:
        return f"Error generating notification: {str(e)}"

def show_product_details(product, df):
    if product.empty:
        return
    st.markdown(f"""
    <div class="product-highlight">
        <h3>{product['name'].iloc[0]}</h3>
        <p><strong>Category:</strong> {product['category'].iloc[0]}</p>
        <p><strong>Current Price:</strong> ₹{product['price'].iloc[0]:,.2f}</p>
        <p><strong>Rating:</strong> {product['rating'].iloc[0]} ⭐ ({product['review_count'].iloc[0]} reviews)</p>
        <p><strong>Availability:</strong> {product['availability'].iloc[0]}</p>
        <p>{product['description'].iloc[0]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    history = get_product_history(product['product_id'].iloc[0], df)
    if not history.empty:
        fig = px.line(history, x='snapshot_time', y='price', 
                     title=f"Price History for {product['name'].iloc[0]}",
                     markers=True)
        st.plotly_chart(fig, use_container_width=True)

def main():
    df = load_data()
    if df.empty:
        st.warning("No data available. Please check the data file.")
        return
    
    vectorizer, tfidf_matrix = build_semantic_search(df)
    
    st.title("🤖 Frido Product Intelligence Agent")
    st.markdown("Discover products, track prices, and get exclusive price drop alerts!")
    
    # User Name for Personalization
    user_name = st.text_input("Enter your name for personalized offers", value="Customer")
    
    # Notification Section
    st.subheader("🎯 Exclusive Offers")
    drops = get_price_drop_alerts(df, min_drop=10)
    if not drops.empty:
        for _, drop in drops.head(3).iterrows():  # Show up to 3 notifications
            notification = generate_notification(drop, user_name)
            if not notification.startswith("Error"):
                st.markdown(f"""
                <div class="notification-box">
                    <span>{notification}</span>
                    <button>Shop Now</button>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(notification)
    else:
        st.info("No exclusive offers available right now. Check back soon!")
    
    # Price Drop Alerts
    st.subheader("🔥 Price Drop Alerts")
    min_drop = st.slider("Minimum Price Drop (%)", 5, 50, 10, step=5)
    if st.button("Show Price Drops"):
        drops = get_price_drop_alerts(df, min_drop)
        if drops.empty:
            st.info(f"No products found with price drops ≥ {min_drop}%")
        else:
            for _, drop in drops.iterrows():
                st.markdown(f"""
                <div class="price-drop-alert">
                    <strong>{drop['name']}</strong><br>
                    Old Price: ₹{drop['old_price']:,.2f} → New Price: ₹{drop['new_price']:,.2f}<br>
                    Drop: {drop['drop_pct']:.1f}% (as of {drop['snapshot_time'].date()})
                </div>
                """, unsafe_allow_html=True)
    
    # Semantic Search
    st.subheader("🔍 Product Search")
    query = st.text_input("Search for a product", 
                         placeholder="e.g., 'comfortable wheelchair with good support'")
    
    if query:
        st.markdown(f"<div class='query-response'><strong>Your query:</strong> {query}</div>", 
                   unsafe_allow_html=True)
        
        results = semantic_search(query, vectorizer, tfidf_matrix, df)
        
        if not results.empty:
            for _, group in results.groupby('product_id'):
                product = group.sort_values('snapshot_time', ascending=False).iloc[0:1]
                
                with st.expander(f"{product['name'].iloc[0]} (₹{product['price'].iloc[0]:,.2f})"):
                    show_product_details(product, df)
                    
                    st.subheader("🔄 Similar Products")
                    similar_products = semantic_search(product['name'].iloc[0], vectorizer, tfidf_matrix, df)
                    
                    for _, sim_product in similar_products[similar_products['product_id'] != product['product_id'].iloc[0]].head(3).iterrows():
                        st.markdown(f"""
                        <div class="similar-product">
                            <strong>{sim_product['name']}</strong>
                            <p>Price: ₹{sim_product['price']:,.2f} | Rating: {sim_product['rating']} ⭐</p>
                            <p>{sim_product['category']} | {sim_product['availability']}</p>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.warning("No products found matching your query. Try different keywords.")
    
    # Advanced Filters
    st.sidebar.header("Advanced Product Search")
    category_filter = st.sidebar.selectbox("Filter by Category", ['All'] + sorted(df['category'].unique().tolist()))
    price_range = st.sidebar.slider("Price Range (₹)", 
                                  float(df['price'].min()), 
                                  float(df['price'].max()), 
                                  (float(df['price'].min()), float(df['price'].max())))
    rating_filter = st.sidebar.slider("Minimum Rating", 1.0, 5.0, 3.0)
    
    if st.sidebar.button("Apply Filters"):
        filtered_df = df.copy()
        if category_filter != 'All':
            filtered_df = filtered_df[filtered_df['category'] == category_filter]
        filtered_df = filtered_df[
            (filtered_df['price'] >= price_range[0]) & 
            (filtered_df['price'] <= price_range[1]) &
            (filtered_df['rating'] >= rating_filter)
        ].drop_duplicates('product_id', keep='last')
        
        st.subheader(f"📊 Filtered Products ({len(filtered_df)})")
        
        for _, product in filtered_df.iterrows():
            with st.expander(f"{product['name']} (₹{product['price']:,.2f})"):
                show_product_details(pd.DataFrame([product]), df)

if __name__ == "__main__":
    main()
