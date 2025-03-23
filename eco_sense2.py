import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from PIL import Image
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sqlite3

# Set page config at the very beginning
st.set_page_config(page_title="EcoSense - Waste Reduction Platform", page_icon="🌱", layout="wide")

# Database Setup
conn = sqlite3.connect("ecosense.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS waste_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    material TEXT,
    quantity REAL,
    date TEXT,
    method TEXT
)
""")
conn.commit()

# Simulated user database
users = {"admin": "password123", "user1": "eco123"}

# ML Model initialization
def initialize_ml_model():
    if not os.path.exists('waste_model.joblib'):
        # Create sample training data
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic data
        X_train = np.array([
            np.random.randint(1, 11, n_samples),  # household_size
            np.random.randint(1, 4, n_samples),   # income_level
            np.random.randint(1, 4, n_samples),   # diet_type
            np.random.randint(1, 8, n_samples),   # shopping_frequency
            np.random.randint(1, 11, n_samples)   # recycling_effort
        ]).T
        
        # Generate target values (weekly waste in kg)
        y_train = (
            2.5 * X_train[:, 0] +  # household size impact
            0.8 * X_train[:, 1] +  # income level impact
            -0.5 * X_train[:, 2] + # diet type impact
            -0.3 * X_train[:, 3] + # shopping frequency impact
            -0.4 * X_train[:, 4] + # recycling effort impact
            np.random.normal(0, 0.5, n_samples)  # random noise
        )
        
        # Create and train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Save the trained model
        joblib.dump(model, 'waste_model.joblib')
    
    return joblib.load('waste_model.joblib')

# Custom CSS for styling
st.markdown(
    """
    <style>
    /* Set Background Color */
    [data-testid="stAppViewContainer"] {
        background-color: #E8F5E9;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #2E7D32 !important;
        color: white !important;
    }

    /* Sidebar Text Styling */
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Buttons Styling */
    div.stButton > button {
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        transition: 0.3s;
    }
    
    div.stButton > button:hover {
        background-color: #388E3C;
    }

    /* Headers */
    .st-emotion-cache-10trblm {
        color: #1B5E20;
    }

    /* Card styling */
    .stCard {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Input fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #4CAF50;
    }

    /* Select boxes */
    .stSelectbox > div > div > div {
        border-radius: 8px;
        border: 1px solid #4CAF50;
    }

    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #1B5E20;
        font-weight: bold;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }

    /* Success messages */
    .stSuccess {
        background-color: #C8E6C9;
        color: #1B5E20;
        padding: 10px;
        border-radius: 8px;
    }

    /* Error messages */
    .stError {
        background-color: #FFCDD2;
        color: #B71C1C;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Session State for login
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = ""

def authenticate(username, password):
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return cursor.fetchone() is not None

def register(username, password):
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except:
        return False

def main():
    if not st.session_state["authenticated"]:
        login()
        return
    
    st.sidebar.title("🌿 EcoSense Menu")
    page = st.sidebar.radio("", ["🏠 Home", "📝 Log Waste", "📊 Insights", "🚀 Action Plan", "🏆 Leaderboard", "💬 Forum", "ℹ️ Project Info", "⚡ Waste-to-Energy"])
    
    if page == "🏠 Home":
        home_screen()
    elif page == "📝 Log Waste":
        log_waste()
    elif page == "📊 Insights":
        waste_insights()
    elif page == "🚀 Action Plan":
        action_plan()
    elif page == "🏆 Leaderboard":
        leaderboard()
    elif page == "💬 Forum":
        community_forum()
    elif page == "⚡ Waste-to-Energy":
        waste_to_energy_calculator()
    else:
        project_info()

def login():
    st.title("👋 Welcome to Your Eco Journey!")
    st.write("Join our community of eco-warriors making a difference, one step at a time.")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Start My Journey"):
        if authenticate(username, password):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Oops! Those credentials don't match our records. Want to try again?")
    
    st.markdown("---")
    st.subheader("🌱 First Time Here?")
    st.write("Create your account and join our mission for a cleaner planet!")
    new_user = st.text_input("Pick a Username")
    new_pass = st.text_input("Create a Password", type="password")
    if st.button("Join the Movement"):
        if register(new_user, new_pass):
            st.success("Welcome to the EcoSense family! You're all set to make a difference.")
        else:
            st.error("This username is already making a difference! Try another one.")

def home_screen():
    st.header("🌍 Your Environmental Impact Dashboard")
    st.write(f"Welcome back, {st.session_state['username']}! Let's see the difference you're making.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Your Impact Score", value="28%", delta="5%")
    with col2:
        st.metric(label="Materials Saved", value="167 kg", delta="23 kg")
    with col3:
        st.metric(label="Carbon Footprint Reduced", value="103 kg", delta="-12 kg")

def log_waste():
    st.header("📝 Log Your Waste")
    
    material = st.selectbox("Material Type", ["Food", "Plastic", "Paper", "Glass", "Metal", "Electronics", "Other"])
    quantity = st.number_input("Amount (kg)", min_value=0.1, max_value=50.0, value=0.5, step=0.1)
    entry_date = st.date_input("Entry Date", datetime.now())
    handling = st.selectbox("Handling Method", ["Recycled", "Composted", "Landfill", "Donated", "Repurposed"])
    
    if st.button("💾 Save Entry"):
        cursor.execute("INSERT INTO waste_log (username, material, quantity, date, method) VALUES (?, ?, ?, ?, ?)",
                      (st.session_state["username"], material, quantity, entry_date.strftime('%Y-%m-%d'), handling))
        conn.commit()
        st.success(f"✅ Logged {quantity} kg of {material}")

def waste_insights():
    st.header("📊 Your Sustainability Journey")
    
    # Add time period selector
    span = st.selectbox("Select Time Period", ["7 Days", "30 Days", "90 Days", "1 Year"])
    days = int(span.split()[0])
    
    # Get data from database for the selected period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    query = """
    SELECT date, material, quantity 
    FROM waste_log 
    WHERE username=? 
    AND date BETWEEN ? AND ?
    ORDER BY date
    """
    
    df = pd.read_sql_query(
        query, 
        conn, 
        params=(
            st.session_state["username"],
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
    )
    
    if not df.empty:
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Create pivot table for materials over time
        pivot_df = df.pivot_table(
            index='date',
            columns='material',
            values='quantity',
            aggfunc='sum'
        ).fillna(0)
        
        # Ensure all dates are present
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        pivot_df = pivot_df.reindex(date_range, fill_value=0)
        
        # Display line chart
        st.subheader("📈 Waste Trends")
        st.line_chart(pivot_df)
        
        # Display bar chart for total by material
        st.subheader("📊 Total Waste by Material")
        fig, ax = plt.subplots(figsize=(10, 6))
        total_by_material = df.groupby('material')['quantity'].sum()
        total_by_material.plot(kind='bar', ax=ax)
        plt.title('Total Waste Distribution by Material Type')
        plt.xlabel('Material')
        plt.ylabel('Total Quantity (kg)')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Display summary metrics
        st.subheader("📌 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_waste = df['quantity'].sum()
            st.metric(
                label="Total Waste",
                value=f"{total_waste:.1f} kg",
                delta=f"{total_waste/days:.1f} kg/day"
            )
        
        with col2:
            most_recycled = df[df['material'] == 'Recycled']['quantity'].sum()
            st.metric(
                label="Recycled Materials",
                value=f"{most_recycled:.1f} kg",
                delta=f"{(most_recycled/total_waste*100):.1f}%"
            )
        
        with col3:
            landfill = df[df['material'] == 'Landfill']['quantity'].sum()
            st.metric(
                label="Landfill Waste",
                value=f"{landfill:.1f} kg",
                delta=f"{(landfill/total_waste*100):.1f}%"
            )
        
        # Display detailed data table
        st.subheader("📋 Detailed Records")
        st.dataframe(
            df.sort_values('date', ascending=False),
            column_config={
                "date": "Date",
                "material": "Material",
                "quantity": st.column_config.NumberColumn(
                    "Quantity (kg)",
                    format="%.2f"
                )
            }
        )
        
        # Download option
        st.download_button(
            "⬇️ Download Complete Data",
            df.to_csv(index=False),
            "waste_data.csv",
            "text/csv",
            key='download-csv'
        )
        
        # Display insights
        st.subheader("💡 Insights")
        most_common_material = df.groupby('material')['quantity'].sum().idxmax()
        daily_average = total_waste / days
        
        st.info(f"""
        Key findings for the last {days} days:
        - Your most disposed material is {most_common_material}
        - Daily average waste: {daily_average:.2f} kg
        - Total waste generated: {total_waste:.2f} kg
        """)
        
        # Recommendations based on data
        st.subheader("🎯 Recommendations")
        if most_common_material == "Plastic":
            st.warning("""
            To reduce plastic waste:
            - Use reusable shopping bags
            - Avoid single-use plastics
            - Choose products with minimal packaging
            """)
        elif most_common_material == "Food":
            st.warning("""
            To reduce food waste:
            - Plan meals in advance
            - Store food properly
            - Start composting
            """)
        elif most_common_material == "Paper":
            st.warning("""
            To reduce paper waste:
            - Go digital when possible
            - Use both sides of paper
            - Recycle properly
            """)
    else:
        st.info("No waste logs found for the selected period. Start logging your waste to see insights!")
        
        # Show sample data visualization
        st.subheader("📊 Sample Visualization")
        sample_dates = pd.date_range(end=datetime.now(), periods=days)
        sample_data = pd.DataFrame({
            "Date": sample_dates,
            "Organic": np.random.normal(1.8, 0.6, days),
            "Plastic": np.random.normal(0.9, 0.4, days),
            "Paper": np.random.normal(0.7, 0.3, days)
        })
        sample_data.set_index("Date", inplace=True)
        st.line_chart(sample_data)
        st.caption("This is a sample visualization. Your actual data will appear here once you start logging waste.")

def action_plan():
    st.header("🎯 Your Green Goals")
    st.write("Let's make sustainability a daily habit!")
    st.progress(0.58)
    st.write("""
    Your next eco-challenges:
    1. Switch to reusable shopping bags
    2. Start a mini herb garden
    3. Try plastic-free shopping week
    """)

def leaderboard():
    st.header("🏆 Environmental Champions")
    df = pd.read_sql_query("""
        SELECT username, 
               SUM(quantity) as total_waste,
               COUNT(*) as entries
        FROM waste_log 
        GROUP BY username 
        ORDER BY total_waste DESC
    """, conn)
    
    if not df.empty:
        st.table(df)
    else:
        st.info("No entries yet. Start logging waste to appear on the leaderboard!")

def community_forum():
    st.header("💚 Community Hub")
    st.write("Share your eco-friendly tips and inspire others!")
    
    comment = st.text_area("Got a sustainable life hack to share?")
    if st.button("Share with Community"):
        st.success("Thanks for spreading the green wisdom! Your tip will inspire others.")
    
    st.subheader("🌟 Community Highlights")
    st.write("- 'I started using beeswax wraps instead of plastic wrap - game changer!' - Recent Community Tip")
    st.write("- 'My composting journey reduced my waste by 40%!' - Success Story")
    st.write("- 'Local farmers market shopping eliminated my plastic packaging' - Weekly Inspiration")

def project_info():
    st.header("🌱 About Our Mission")
    st.write("""
    Welcome to EcoSense - where every small action counts towards a bigger change. 
    We're not just another waste tracking platform; we're a community of change-makers 
    committed to creating a more sustainable future.
    """)
    
    st.markdown("""
    ### 🚀 What Makes Us Different
    
    We believe in making sustainability accessible and engaging. Our platform combines:
    - **Smart Tracking**: Understand your environmental impact in real-time
    - **Community Power**: Learn and share with fellow eco-warriors
    - **Personalized Journey**: Get tips and insights tailored to your habits
    - **Visual Progress**: Watch your positive impact grow over time
    - **Real Results**: Turn your daily actions into measurable change
    
    ### 🤝 Join Our Community
    - **GitHub:** [Check out our open-source project](https://github.com/Aayush5154/Ecosense2)
    - **Share Ideas:** [Contribute to development](https://github.com/Aayush5154/Ecosense2/issues)
    - **Learn More:** [Read our story](https://github.com/Aayush5154/Ecosense2#readme)
    
    ### 📈 Platform Status
    Current Version: 1.0.0 - Building a greener future together
    """)

def waste_to_energy_calculator():
    st.header("⚡ Waste-to-Energy Calculator")
    st.write("Discover how your waste could be transformed into renewable energy!")
    
    # Input section
    st.subheader("📊 Your Waste Profile")
    col1, col2 = st.columns(2)
    
    with col1:
        organic_waste = st.slider("Organic Waste (kg/week)", 0.0, 20.0, 5.0, 0.1)
        plastic_waste = st.slider("Plastic Waste (kg/week)", 0.0, 10.0, 2.0, 0.1)
        paper_waste = st.slider("Paper Waste (kg/week)", 0.0, 10.0, 1.5, 0.1)
    
    with col2:
        wood_waste = st.slider("Wood Waste (kg/week)", 0.0, 10.0, 1.0, 0.1)
        textile_waste = st.slider("Textile Waste (kg/week)", 0.0, 5.0, 0.5, 0.1)
        other_waste = st.slider("Other Waste (kg/week)", 0.0, 5.0, 0.5, 0.1)
    
    # Energy conversion factors (kWh/kg)
    conversion_factors = {
        'Organic': 0.5,    # Biogas potential
        'Plastic': 8.0,    # Incineration energy
        'Paper': 4.0,      # Incineration energy
        'Wood': 3.5,       # Biomass energy
        'Textile': 5.0,    # Incineration energy
        'Other': 2.0       # General waste
    }
    
    # Calculate energy potential
    weekly_waste = {
        'Organic': organic_waste,
        'Plastic': plastic_waste,
        'Paper': paper_waste,
        'Wood': wood_waste,
        'Textile': textile_waste,
        'Other': other_waste
    }
    
    # Calculate energy potential for each waste type
    energy_potential = {
        waste_type: amount * conversion_factors[waste_type]
        for waste_type, amount in weekly_waste.items()
    }
    
    total_energy = sum(energy_potential.values())
    
    # Display results
    st.subheader("⚡ Energy Potential")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Weekly Energy", f"{total_energy:.1f} kWh",
                 f"Enough to power {total_energy/10:.1f} homes for a day")
    with col2:
        st.metric("Monthly Energy", f"{total_energy * 4:.1f} kWh",
                 f"Equivalent to {total_energy * 4 / 100:.1f} trees planted")
    with col3:
        st.metric("Yearly Energy", f"{total_energy * 52:.1f} kWh",
                 f"CO₂ reduction: {total_energy * 52 * 0.7:.1f} kg")
    
    # Visualize energy potential
    st.subheader("🌱 Energy Potential by Waste Type")
    
    energy_data = pd.DataFrame({
        'Waste Type': list(energy_potential.keys()),
        'Energy (kWh)': list(energy_potential.values())
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(energy_data['Waste Type'], energy_data['Energy (kWh)'])
    ax.set_ylabel('Energy Potential (kWh)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    st.pyplot(fig)
    
    # Educational content
    st.subheader("📚 Learn More About Waste-to-Energy")
    
    with st.expander("How does waste-to-energy work?"):
        st.write("""
        Waste-to-energy processes convert waste materials into usable forms of energy:
        - Organic waste → Biogas through anaerobic digestion
        - Plastic/Paper → Energy through controlled incineration
        - Wood → Biomass energy
        - Textiles → Energy recovery or recycling
        """)
    
    with st.expander("Environmental Impact"):
        st.write("""
        Converting waste to energy:
        - Reduces landfill waste
        - Generates renewable energy
        - Reduces greenhouse gas emissions
        - Creates circular economy opportunities
        """)
    
    # Action recommendations
    st.subheader("💡 Recommendations")
    
    if organic_waste > 5:
        st.info("🌱 Consider starting a home composting system to convert organic waste into valuable fertilizer!")
    
    if plastic_waste > 3:
        st.info("♻️ Look into local plastic recycling programs to reduce waste and energy consumption.")
    
    if paper_waste > 2:
        st.info("📚 Switch to digital documents where possible to reduce paper waste.")
    
    # Share results
    if st.button("📢 Share Your Energy Potential"):
        st.success("✅ Your waste-to-energy potential has been shared with the community!")
        st.balloons()

if __name__ == "__main__":
    main()
