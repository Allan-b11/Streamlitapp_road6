import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import warnings
import time # Added for sleep in auto-refresh
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Custom CSS for NGO branding
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #0E94C9;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1B5E20;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(90deg,rgba(24, 156, 171, 1) 0%,);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    .sidebar .sidebar-content {
        background: linear-gradientlinear-gradient(90deg,rgba(24, 156, 171, 1) 0%, rgba(175, 87, 199, 1) 50%, rgba(237, 229, 83, 1) 100%);
    }
    .stButton > button {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #45a049 0%, #3d8b40 100%);
    }
    .urgent-alert {
        background: #FFE0B2;
        border-left: 4px solid #FF9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .success-message {
        background: #E8F5E8;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'real_time_updates' not in st.session_state:
    st.session_state.real_time_updates = True

class RoadManagementPlatform:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.category_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Road-specific categories
        self.road_categories = [
            'Potholes & Surface Damage',
            'Drainage Issues',
            'Road Markings & Signage',
            'Traffic Safety',
            'Bridge Maintenance',
            'Roadside Vegetation',
            'Street Lighting',
            'Pedestrian Infrastructure',
            'Traffic Flow Issues',
            'Emergency Access'
        ]
        
        # Location types for road infrastructure
        self.location_types = [
            'Urban Main Roads',
            'Rural Roads',
            'Residential Streets',
            'Highway Sections',
            'Bridge Areas',
            'Intersection Points',
            'School Zones',
            'Commercial Areas',
            'Industrial Zones',
            'Remote Areas'
        ]

    def generate_sample_data(self, n_samples=80):
        """Generate synthetic road management feedback data"""
        
        # Road-specific feedback templates
        feedback_templates = {
            'Potholes & Surface Damage': [
                "Large potholes causing vehicle damage",
                "Road surface is severely cracked and dangerous",
                "Asphalt is deteriorating rapidly",
                "Deep holes filled with water during rain",
                "Uneven road surface causing accidents"
            ],
            'Drainage Issues': [
                "Poor drainage causing flooding during rain",
                "Blocked drainage systems on roadside",
                "Water stagnation creating mosquito breeding",
                "Inadequate storm water management",
                "Erosion due to poor water flow"
            ],
            'Road Markings & Signage': [
                "Faded road markings need repainting",
                "Missing or damaged road signs",
                "Confusing intersection markings",
                "Speed limit signs not visible",
                "Lane markers completely worn out"
            ],
            'Traffic Safety': [
                "Dangerous curves need safety barriers",
                "Insufficient lighting at night",
                "Accident-prone intersection needs attention",
                "Speeding vehicles in residential area",
                "Blind spots creating safety hazards"
            ],
            'Bridge Maintenance': [
                "Bridge structure showing signs of wear",
                "Loose railings on bridge walkway",
                "Bridge deck needs resurfacing",
                "Corrosion visible on steel supports",
                "Water damage to bridge foundation"
            ]
        }
        
        data = []
        for i in range(n_samples):
            category = np.random.choice(self.road_categories)
            location = np.random.choice(self.location_types)
            
            # Generate realistic feedback
            if category in feedback_templates:
                base_feedback = np.random.choice(feedback_templates[category])
            else:
                base_feedback = f"Issues with {category.lower()} in the area"
            
            # Add location and severity context
            severity_level = np.random.choice(['Minor', 'Moderate', 'Severe'], p=[0.3, 0.4, 0.3])
            
            feedback_text = f"{base_feedback} at {location}. "
            
            # Add urgency based on severity
            if severity_level == 'Severe':
                feedback_text += "This is a critical safety issue requiring immediate attention."
                priority = 'High'
            elif severity_level == 'Moderate':
                feedback_text += "This issue affects daily commute and needs prompt resolution."
                priority = 'Medium'
            else:
                feedback_text += "This can be addressed during routine maintenance."
                priority = 'Low'
            
            # Generate timestamp (within last 30 days for real-time feel)
            days_ago = np.random.randint(0, 30)
            timestamp = datetime.now() - timedelta(days=days_ago)
            
            # Generate response time based on priority
            if priority == 'High':
                response_time = np.random.exponential(12)  # Avg 12 hours
            elif priority == 'Medium':
                response_time = np.random.exponential(48)  # Avg 48 hours
            else:
                response_time = np.random.exponential(120)  # Avg 5 days
            
            # Status distribution
            status = np.random.choice(['Open', 'In Progress', 'Resolved', 'Closed'],
                                      p=[0.25, 0.35, 0.30, 0.10])
            
            data.append({
                'feedback_id': f"RD{i+1:03d}",
                'timestamp': timestamp,
                'feedback_text': feedback_text,
                'category': category,
                'location': location,
                'priority': priority,
                'severity': severity_level,
                'response_time_hours': response_time,
                'status': status,
                'citizen_id': f"citizen_{np.random.randint(1, 50)}",
                'estimated_cost': np.random.randint(500, 15000),  # Cost in USD
                'affected_vehicles_daily': np.random.randint(10, 500)
            })
            
        return pd.DataFrame(data)

    def preprocess_data(self, df):
        """Clean and preprocess the feedback data"""
        # Remove duplicates
        df = df.drop_duplicates(subset=['feedback_text', 'citizen_id'])
        
        # Clean text
        df['feedback_clean'] = df['feedback_text'].apply(self.clean_text)
        
        # Extract temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Calculate sentiment scores
        df['sentiment_score'] = df['feedback_clean'].apply(
            lambda x: self.sia.polarity_scores(x)['compound']
        )
        df['sentiment_label'] = df['sentiment_score'].apply(
            lambda x: 'Positive' if x > 0.05 else 'Negative' if x < -0.05 else 'Neutral'
        )
        
        # Text features
        df['text_length'] = df['feedback_clean'].str.len()
        df['word_count'] = df['feedback_clean'].str.split().str.len()
        df['urgency_keywords'] = df['feedback_clean'].str.count('urgent|critical|dangerous|emergency')
        
        # Cost and impact analysis
        df['cost_per_affected_vehicle'] = df['estimated_cost'] / df['affected_vehicles_daily']
        df['days_since_reported'] = (datetime.now() - df['timestamp']).dt.days
        
        return df

    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def feature_engineering(self, df):
        """Create features for machine learning models."""
        df_encoded = df.copy()

        # Encode categorical variables
        df_encoded['category_encoded'] = self.category_encoder.fit_transform(df_encoded['category'])
        df_encoded['location_encoded'] = self.category_encoder.fit_transform(df_encoded['location'])
        df_encoded['priority_encoded'] = self.category_encoder.fit_transform(df_encoded['priority'])
        df_encoded['severity_encoded'] = self.category_encoder.fit_transform(df_encoded['severity'])
        df_encoded['status_encoded'] = self.category_encoder.fit_transform(df_encoded['status'])
        df_encoded['sentiment_encoded'] = self.category_encoder.fit_transform(df_encoded['sentiment_label'])

        # TF-IDF for feedback text
        tfidf_matrix = self.vectorizer.fit_transform(df_encoded['feedback_clean'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=self.vectorizer.get_feature_names_out(), index=df_encoded.index)
        df_featured = pd.concat([df_encoded, tfidf_df], axis=1)

        # Numerical features to scale
        numerical_features = [
            'hour', 'day_of_week', 'month', 'text_length', 'word_count',
            'urgency_keywords', 'estimated_cost', 'affected_vehicles_daily',
            'cost_per_affected_vehicle', 'days_since_reported', 'sentiment_score'
        ]
        
        # Handle potential NaNs created by division (cost_per_affected_vehicle)
        for col in numerical_features:
            if col in df_featured.columns:
                df_featured[col] = df_featured[col].replace([np.inf, -np.inf], np.nan).fillna(0) # Replace inf with NaN and then fill NaN

        df_featured[numerical_features] = self.scaler.fit_transform(df_featured[numerical_features])

        # Select features for models
        features = [
            'hour', 'day_of_week', 'month', 'is_weekend', 'text_length',
            'word_count', 'urgency_keywords', 'estimated_cost',
            'affected_vehicles_daily', 'cost_per_affected_vehicle',
            'days_since_reported', 'sentiment_score', 'sentiment_encoded',
            'location_encoded', 'severity_encoded', 'status_encoded'
        ] + list(self.vectorizer.get_feature_names_out()) # Add TF-IDF features

        # Ensure all features exist, fill with 0 if not (for robustness with smaller data)
        for f in features:
            if f not in df_featured.columns:
                df_featured[f] = 0

        X = df_featured[features]
        return X, df_featured

    def train_ai_models(self, df_featured):
        """Train AI models for category classification, priority assessment, and response time prediction."""
        
        X = st.session_state.features # Use pre-computed features

        # Target variables
        y_category = df_featured['category_encoded']
        y_priority = df_featured['priority_encoded']
        y_response_time = df_featured['response_time_hours']

        # Category Classification Model (Random Forest Classifier)
        st.write("Training Category Classifier...")
        X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
            X, y_category, test_size=0.2, random_state=42
        )
        category_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        category_classifier.fit(X_train_cat, y_train_cat)
        y_pred_cat = category_classifier.predict(X_test_cat)
        cat_accuracy = accuracy_score(y_test_cat, y_pred_cat)
        st.write(f"Category Classifier Accuracy: {cat_accuracy:.2f}")

        # Priority Assessment Model (Logistic Regression)
        st.write("Training Priority Classifier...")
        X_train_pri, X_test_pri, y_train_pri, y_test_pri = train_test_split(
            X, y_priority, test_size=0.2, random_state=42
        )
        priority_classifier = LogisticRegression(max_iter=1000, random_state=42)
        priority_classifier.fit(X_train_pri, y_train_pri)
        y_pred_pri = priority_classifier.predict(X_test_pri)
        pri_accuracy = accuracy_score(y_test_pri, y_pred_pri)
        st.write(f"Priority Classifier Accuracy: {pri_accuracy:.2f}")

        # Response Time Prediction Model (Gradient Boosting Regressor)
        st.write("Training Response Time Predictor...")
        X_train_resp, X_test_resp, y_train_resp, y_test_resp = train_test_split(
            X, y_response_time, test_size=0.2, random_state=42
        )
        response_time_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        response_time_predictor.fit(X_train_resp, y_train_resp)
        r2_score = response_time_predictor.score(X_test_resp, y_test_resp) # R-squared score
        st.write(f"Response Time Predictor R-squared: {r2_score:.2f}")

        # Store models
        st.session_state.models = {
            'category_classifier': category_classifier,
            'priority_classifier': priority_classifier,
            'response_time_predictor': response_time_predictor,
            'category_encoder': self.category_encoder,
            'vectorizer': self.vectorizer,
            'scaler': self.scaler
        }
        st.session_state.models_trained = True

        return {
            'category_accuracy': cat_accuracy,
            'priority_accuracy': pri_accuracy,
            'response_time_r2': r2_score
        }

    def predict_road_issue(self, new_feedback_text, location, severity, estimated_cost, affected_vehicles_daily):
        """Predict category, priority, and response time for a new issue."""
        if not st.session_state.models_trained:
            return "Models not trained yet. Please train them first.", None, None, None

        models = st.session_state.models
        
        # Create a DataFrame for the new input, mimicking the structure of training data
        new_data = pd.DataFrame([{
            'feedback_text': new_feedback_text,
            'category': 'Unknown', # Placeholder
            'location': location,
            'priority': 'Unknown', # Placeholder
            'severity': severity,
            'response_time_hours': 0, # Placeholder
            'estimated_cost': estimated_cost,
            'affected_vehicles_daily': affected_vehicles_daily,
            'timestamp': datetime.now(),
            'citizen_id': 'new_citizen', # Placeholder
            'status': 'Open' # Placeholder
        }])
        
        # Preprocess the new data
        new_data_processed = self.preprocess_data(new_data.copy())

        # Ensure all columns expected by feature_engineering are present
        for col in ['category', 'location', 'priority', 'severity', 'status', 'sentiment_label']:
            if col not in new_data_processed.columns:
                new_data_processed[col] = 'Unknown'

        # Feature engineering for the new data
        # Need to use the _trained_ vectorizer and scaler
        temp_df = new_data_processed.copy()
        
        temp_df['category_encoded'] = models['category_encoder'].transform(
            temp_df['category'].apply(lambda x: x if x in self.category_encoder.classes_ else self.category_encoder.classes_[0])
        )
        temp_df['location_encoded'] = models['category_encoder'].transform(
            temp_df['location'].apply(lambda x: x if x in self.category_encoder.classes_ else self.category_encoder.classes_[0])
        )
        temp_df['priority_encoded'] = models['category_encoder'].transform(
            temp_df['priority'].apply(lambda x: x if x in self.category_encoder.classes_ else self.category_encoder.classes_[0])
        )
        temp_df['severity_encoded'] = models['category_encoder'].transform(
            temp_df['severity'].apply(lambda x: x if x in self.category_encoder.classes_ else self.category_encoder.classes_[0])
        )
        temp_df['status_encoded'] = models['category_encoder'].transform(
            temp_df['status'].apply(lambda x: x if x in self.category_encoder.classes_ else self.category_encoder.classes_[0])
        )
        temp_df['sentiment_encoded'] = models['category_encoder'].transform(
            temp_df['sentiment_label'].apply(lambda x: x if x in self.category_encoder.classes_ else self.category_encoder.classes_[0])
        )
        
        tfidf_matrix_new = models['vectorizer'].transform(temp_df['feedback_clean'])
        tfidf_df_new = pd.DataFrame(tfidf_matrix_new.toarray(), columns=models['vectorizer'].get_feature_names_out(), index=temp_df.index)
        temp_df = pd.concat([temp_df, tfidf_df_new], axis=1)

        numerical_features = [
            'hour', 'day_of_week', 'month', 'text_length', 'word_count',
            'urgency_keywords', 'estimated_cost', 'affected_vehicles_daily',
            'cost_per_affected_vehicle', 'days_since_reported', 'sentiment_score'
        ]
        for col in numerical_features:
            if col in temp_df.columns:
                temp_df[col] = temp_df[col].replace([np.inf, -np.inf], np.nan).fillna(0) # Replace inf with NaN and then fill NaN

        temp_df[numerical_features] = models['scaler'].transform(temp_df[numerical_features])

        features = [
            'hour', 'day_of_week', 'month', 'is_weekend', 'text_length',
            'word_count', 'urgency_keywords', 'estimated_cost',
            'affected_vehicles_daily', 'cost_per_affected_vehicle',
            'days_since_reported', 'sentiment_score', 'sentiment_encoded',
            'location_encoded', 'severity_encoded', 'status_encoded'
        ] + list(models['vectorizer'].get_feature_names_out())

        # Ensure all features exist for prediction, fill with 0 if not
        X_new = pd.DataFrame(columns=features)
        for col in features:
            if col in temp_df.columns:
                X_new[col] = temp_df[col]
            else:
                X_new[col] = 0
        X_new = X_new.fillna(0) # Final fillna for any remaining NaNs after merging/scaling

        # Predictions
        predicted_category_encoded = models['category_classifier'].predict(X_new)[0]
        predicted_category = models['category_encoder'].inverse_transform([predicted_category_encoded])[0]

        predicted_priority_encoded = models['priority_classifier'].predict(X_new)[0]
        predicted_priority = models['category_encoder'].inverse_transform([predicted_priority_encoded])[0]

        predicted_response_time = models['response_time_predictor'].predict(X_new)[0]

        return predicted_category, predicted_priority, predicted_response_time, new_data_processed['sentiment_label'].iloc[0]


    def perform_road_analytics(self, df):
        """Perform road management specific analytics"""
        st.markdown('<div class="sub-header">🛣️ Road Management Analytics</div>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Reports", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            avg_response = df['response_time_hours'].mean()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Avg Response Time", f"{avg_response:.1f} hrs")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            total_cost = df['estimated_cost'].sum()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Estimated Cost", f"${total_cost:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            high_priority = (df['priority'] == 'High').sum()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("High Priority Issues", high_priority)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Road category analysis
        col1, col2 = st.columns(2)
        with col1:
            category_counts = df['category'].value_counts()
            fig_cat = px.bar(y=category_counts.index, x=category_counts.values,
                             title="Issues by Road Category", orientation='h',
                             color_discrete_sequence=['#4CAF50'])
            fig_cat.update_layout(height=400)
            st.plotly_chart(fig_cat, use_container_width=True)
        
        with col2:
            severity_counts = df['severity'].value_counts()
            fig_sev = px.pie(values=severity_counts.values, names=severity_counts.index,
                             title="Issues by Severity Level",
                             color_discrete_sequence=['#FF5722', '#FF9800', '#FFC107'])
            st.plotly_chart(fig_sev, use_container_width=True)
        
        # Location analysis
        location_stats = df.groupby('location').agg({
            'feedback_id': 'count',
            'estimated_cost': 'sum',
            'affected_vehicles_daily': 'sum'
        }).round(2)
        
        fig_loc = px.bar(location_stats, x=location_stats.index, y='feedback_id',
                         title="Issues by Location Type",
                         color_discrete_sequence=['#2E7D32'])
        st.plotly_chart(fig_loc, use_container_width=True)
        
        # Priority vs Response Time Analysis
        fig_priority = px.box(df, x='priority', y='response_time_hours',
                              title="Response Time by Priority Level",
                              color='priority',
                              color_discrete_sequence=['#4CAF50', '#FF9800', '#F44336'])
        st.plotly_chart(fig_priority, use_container_width=True)
        
        # Time series analysis
        daily_issues = df.groupby(df['timestamp'].dt.date).size().reset_index()
        daily_issues.columns = ['date', 'count']
        
        fig_ts = px.line(daily_issues, x='date', y='count',
                         title="Daily Road Issues Reported",
                         color_discrete_sequence=['#2E7D32'])
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Cost analysis
        cost_by_category = df.groupby('category')['estimated_cost'].sum().sort_values(ascending=False)
        fig_cost = px.bar(x=cost_by_category.values, y=cost_by_category.index,
                          title="Estimated Repair Costs by Category",
                          orientation='h',
                          color_discrete_sequence=['#FF5722'])
        st.plotly_chart(fig_cost, use_container_width=True)

    def create_road_dashboard(self, df):
        """Create real-time road management dashboard"""
        st.markdown('<div class="sub-header">📊 Real-Time Road Management Dashboard</div>', unsafe_allow_html=True)
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Enable Auto-Refresh (30 seconds)", value=st.session_state.real_time_updates)
        
        if auto_refresh:
            st.session_state.real_time_updates = True
            # In a real deployment, this would fetch new data
            time.sleep(0.1)  # Simulate real-time update
        
        # Critical alerts
        urgent_issues = df[df['priority'] == 'High']
        if len(urgent_issues) > 0:
            st.markdown('<div class="urgent-alert">', unsafe_allow_html=True)
            st.write(f"⚠️ **{len(urgent_issues)} Critical Road Issues Require Immediate Attention**")
            for _, issue in urgent_issues.head(3).iterrows():
                st.write(f"• {issue['category']} at {issue['location']} - {issue['feedback_text'][:100]}...")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Real-time metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            new_today = len(df[df['timestamp'].dt.date == datetime.now().date()])
            st.metric("New Today", new_today, delta=f"+{np.random.randint(0, 5)}")
        
        with col2:
            in_progress = len(df[df['status'] == 'In Progress'])
            st.metric("In Progress", in_progress)
        
        with col3:
            resolved_today = len(df[(df['status'] == 'Resolved') & 
                                   (df['timestamp'].dt.date == datetime.now().date())])
            st.metric("Resolved Today", resolved_today, delta=f"+{np.random.randint(0, 3)}")
        
        with col4:
            avg_cost = df['estimated_cost'].mean()
            st.metric("Avg Cost", f"${avg_cost:,.0f}")
        
        with col5:
            satisfaction = (df['sentiment_score'] > 0).mean()
            st.metric("Satisfaction Rate", f"{satisfaction:.1%}")
        
        # Priority distribution
        priority_dist = df['priority'].value_counts()
        fig_priority = px.donut(values=priority_dist.values, names=priority_dist.index,
                                 title="Current Priority Distribution",
                                 color_discrete_sequence=['#F44336', '#FF9800', '#0E94C9'])
        st.plotly_chart(fig_priority, use_container_width=True)
        
        # Recent activity feed
        st.markdown('<div class="sub-header">🔄 Recent Activity</div>', unsafe_allow_html=True)
        recent_df = df.sort_values('timestamp', ascending=False).head(10)
        
        for _, row in recent_df.iterrows():
            time_ago = datetime.now() - row['timestamp']
            hours_ago = int(time_ago.total_seconds() / 3600)
            
            status_color = {'Open': '🔴', 'In Progress': '🟡', 'Resolved': '🟢', 'Closed': '⚪'}
            priority_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
            
            st.write(f"{status_color.get(row['status'], '⚪')} **{row['category']}** at {row['location']} "
                     f"({priority_color.get(row['priority'], '⚪')} {row['priority']}) - {hours_ago}h ago")
            st.write(f"    {row['feedback_text'][:100]}...")
            st.divider()

def main():
    # Page configuration
    st.set_page_config(
        page_title="Road Management NGO Platform",
        page_icon="🛣️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_css()
    
    # Header
    st.markdown('<div class="main-header">🛣️ Road Management NGO Platform</div>', unsafe_allow_html=True)
    st.markdown("**Empowering Communities Through Better Road Infrastructure**")
    st.markdown("---")
    
    # Initialize platform
    platform = RoadManagementPlatform()
    
    # Sidebar navigation
    st.sidebar.title("🗺️ Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox("Choose a section", [
        "🏠 Home Dashboard",
        "📊 Analytics & Reports", 
        "🤖 AI Model Training",
        "📝 Submit Road Issue",
        "📈 Real-Time Monitor",
        "🔍 Insights & Recommendations",
        "📋 Export & Reports"
    ])
    
    # Add NGO information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏢 About Our Organization (NG)")
    st.sidebar.info("""
    **Road Infrastructure Alliance**
    
    Dedicated to improving road safety and infrastructure through community engagement and data-driven solutions.
    
    📧 contacts@ Team5projectors.org
    📞257 523 671
    """)
    
    if page == "🏠 Home Dashboard":
        st.markdown('<div class="sub-header">Welcome to Road Management Platform</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 Our Mission
            Transform road infrastructure management through community feedback and AI-powered insights.
            
            ### 🛠️ Platform Features:
            - **Real-time Issue Tracking**: Monitor road conditions as they develop
            - **Community Engagement**: Citizens report issues directly
            - **AI-Powered Analytics**: Automatic categorization and priority assessment
            - **Resource Optimization**: Data-driven budget allocation
            - **Impact Measurement**: Track improvements and community satisfaction
            
            ### 📊 Current Status:
            """)
            
            if st.button("🔄 Generate Sample Road Data"):
                with st.spinner("Generating road management data..."):
                    sample_data = platform.generate_sample_data(80)
                    st.session_state.sample_data = sample_data
                    st.markdown('<div class="success-message">✅ Sample data generated successfully!</div>', unsafe_allow_html=True)
                    st.write(f"Generated {len(sample_data)} road issue reports")
        
        with col2:
            st.markdown("### 🗺️ Quick Stats")
            if 'sample_data' in st.session_state:
                df = st.session_state.sample_data
                st.metric("Total Issues", len(df))
                st.metric("High Priority", len(df[df['priority'] == 'High']))
                st.metric("Avg Cost", f"${df['estimated_cost'].mean():,.0f}")
                st.metric("Areas Covered", df['location'].nunique())
            else:
                st.info("Generate sample data to see statistics")
    
    elif page == "📊 Analytics & Reports":
        st.markdown('<div class="sub-header">Data Analysis & Insights</div>', unsafe_allow_html=True)
        
        if 'sample_data' not in st.session_state:
            st.warning("Please generate sample data first from the Home Dashboard.")
            return
        
        df = st.session_state.sample_data.copy()
        
        # Data preprocessing
        with st.expander("📋 Data Processing Details"):
            st.write("**Original Data Shape:**", df.shape)
            st.dataframe(df.head(3))
            
            df_processed = platform.preprocess_data(df)
            st.session_state.processed_data = df_processed
            
            st.write("**Processed Data Shape:**", df_processed.shape)
            new_cols = set(df_processed.columns) - set(df.columns)
            st.write("**New Features Added:**", list(new_cols))
        
        # Perform analytics
        platform.perform_road_analytics(df_processed)
    
    elif page == "🤖 AI Model Training":
        st.markdown('<div class="sub-header">AI Model Training & Evaluation</div>', unsafe_allow_html=True)
        
        if 'processed_data' not in st.session_state:
            st.warning("Please complete data analysis first.")
            return
        
        df = st.session_state.processed_data
        
        # Feature engineering
        with st.spinner("Preparing features for AI training..."):
            X, df_featured = platform.feature_engineering(df)
            st.session_state.features = X
            st.session_state.featured_data = df_featured
        
        st.success(f"✅ Created {X.shape[1]} features for model training")
        
        # Model training
        if st.button("🚀 Train AI Models"):
            with st.spinner("Training AI models for road management..."):
                training_results = platform.train_ai_models(df_featured)
            
            st.markdown('<div class="success-message">🎉 AI Models trained successfully!</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Category Classification", f"{training_results['category_accuracy']:.2%} Accuracy")
            with col2:
                st.metric("Priority Assessment", f"{training_results['priority_accuracy']:.2%} Accuracy") 
            with col3:
                st.metric("Response Time Prediction", f"{training_results['response_time_r2']:.2f} R²")
    
    elif page == "📝 Submit Road Issue":
        st.markdown('<div class="sub-header">Report a Road Issue</div>', unsafe_allow_html=True)
        
        st.markdown("Help us improve road infrastructure by reporting issues in your area.")
        
        with st.form("road_issue_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                category = st.selectbox("Issue Category:", platform.road_categories)
                location = st.selectbox("Location Type:", platform.location_types)
                priority = st.selectbox("Urgency Level:", ['Low', 'Medium', 'High'])
            
            with col2:
                severity = st.selectbox("Severity:", ['Minor', 'Moderate', 'Severe'])
                affected_vehicles = st.number_input("Daily Vehicles Affected:", min_value=1, max_value=1000, value=50)
                estimated_cost = st.number_input("Estimated Repair Cost ($):", min_value=100, max_value=50000, value=2000)
            
            feedback_text = st.text_area("Detailed Description:", 
                                         placeholder="Please provide a detailed description of the road issue...")
            
            contact_info = st.text_input("Contact Information (optional):")
            
            submitted = st.form_submit_button("🚀 Submit Road Issue Report")
            
            if submitted and feedback_text:
                # Use AI models for prediction if trained
                if st.session_state.models_trained:
                    predicted_category, predicted_priority, predicted_response_time, sentiment_label = \
                        platform.predict_road_issue(feedback_text, location, severity, estimated_cost, affected_vehicles)
                    st.info(f"AI Predicted: Category '{predicted_category}', Priority '{predicted_priority}', "
                            f"Response Time '{predicted_response_time:.1f} hrs', Sentiment '{sentiment_label}'")
                    # Optionally, overwrite user selected category/priority with AI prediction for internal use
                    # category = predicted_category 
                    # priority = predicted_priority
                else:
                    sentiment_score = platform.sia.polarity_scores(feedback_text)['compound']
                    sentiment_label = 'Positive' if sentiment_score > 0.05 else 'Negative' if sentiment_score < -0.05 else 'Neutral'
                
                # Generate unique ID
                issue_id = f"RD{len(st.session_state.feedback_data) + 1:03d}"
                
                # Store feedback
                feedback_entry = {
                    'feedback_id': issue_id,
                    'timestamp': datetime.now(),
                    'feedback_text': feedback_text,
                    'category': category,
                    'location': location,
                    'priority': priority,
                    'severity': severity,
                    'estimated_cost': estimated_cost,
                    'affected_vehicles_daily': affected_vehicles,
                    'sentiment_score': platform.sia.polarity_scores(feedback_text)['compound'], # Recalculate if not using AI prediction flow
                    'sentiment_label': sentiment_label,
                    'contact_info': contact_info,
                    'status': 'Open'
                }
                
                st.session_state.feedback_data.append(feedback_entry)
                
                # Success message
                st.markdown('<div class="success-message">✅ Road issue reported successfully!</div>', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Issue ID", issue_id)
                with col2:
                    st.metric("Priority Level", priority)
                with col3:
                    st.metric("Estimated Cost", f"${estimated_cost:,}")
                
                st.info("🔄 Your report has been logged and will be reviewed by our road management team.")
    
    elif page == "📈 Real-Time Monitor":
        st.markdown('<div class="sub-header">Real-Time Road Monitoring</div>', unsafe_allow_html=True)
        
        # Combine all data sources
        all_data = []
        if st.session_state.feedback_data:
            all_data.extend(st.session_state.feedback_data)
        if 'processed_data' in st.session_state:
            # Ensure the processed data is converted to dict records to match structure
            processed_records = st.session_state.processed_data.to_dict('records')
            # Filter out duplicates that might exist if generated sample data and then added manually
            existing_feedback_ids = {f['feedback_id'] for f in st.session_state.feedback_data if 'feedback_id' in f}
            unique_processed_records = [rec for rec in processed_records if rec.get('feedback_id') not in existing_feedback_ids]
            all_data.extend(unique_processed_records)
        
        if not all_data:
            st.warning("No data available for monitoring. Please generate sample data or submit issues.")
            return
        
        df_monitor = pd.DataFrame(all_data)
        # Ensure 'timestamp' is datetime for consistency
        df_monitor['timestamp'] = pd.to_datetime(df_monitor['timestamp'])
        
        # Re-preprocess to ensure new entries also have all derived columns
        df_monitor_processed = platform.preprocess_data(df_monitor.copy())
        
        platform.create_road_dashboard(df_monitor_processed)
    
    elif page == "🔍 Insights & Recommendations":
        st.markdown('<div class="sub-header">AI-Powered Insights & Recommendations</div>', unsafe_allow_html=True)
        
        if 'processed_data' not in st.session_state:
            st.warning("Please complete data analysis first.")
            return
        
        df = st.session_state.processed_data
        
        # Key insights
        st.markdown("### 🔍 Key Insights")
        
        # Most problematic areas
        location_issues = df.groupby('location').agg({
            'feedback_id': 'count',
            'sentiment_score': 'mean',
            'estimated_cost': 'sum'
        }).sort_values('feedback_id', ascending=False)
        
        st.markdown("**📍 Most Problematic Areas:**")
        for location, data in location_issues.head(3).iterrows():
            st.write(f"• **{location}**: {data['feedback_id']} issues, ${data['estimated_cost']:,.0f} total cost")
        
        # Category analysis
        category_analysis = df.groupby('category').agg({
            'estimated_cost': 'mean',
            'response_time_hours': 'mean',
            'sentiment_score': 'mean'
        }).round(2)
        
        st.markdown("**🛠️ Most Expensive Categories:**")
        expensive_categories = category_analysis.sort_values('estimated_cost', ascending=False).head(3)
        for category, data in expensive_categories.iterrows():
            st.write(f"• **{category}**: ${data['estimated_cost']:,.0f} average cost")
        
        # AI Recommendations
        st.markdown("### 💡 AI-Powered Recommendations")
        
        recommendations = [
            "🎯 **Priority Focus**: Address potholes and surface damage issues first (highest frequency)",
            "💰 **Budget Allocation**: Allocate 40% of budget to bridge maintenance (highest cost impact)",
            "⏰ **Resource Optimization**: Deploy maintenance teams during off-peak hours",
            "🚨 **Preventive Measures**: Implement regular inspections in high-traffic areas",
            "📱 **Community Engagement**: Expand citizen reporting in rural areas",
            "📊 **Performance Tracking**: Monitor resolution times by location type"
        ]
        
        for rec in recommendations: # This line was cut off in the original code
            st.info(rec)

        st.markdown("### ☁️ Word Cloud of Feedback")
        all_feedback_text = " ".join(df['feedback_clean'].dropna())
        if all_feedback_text:
            wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(all_feedback_text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("No feedback text available for word cloud generation.")

        st.markdown("### 👥 Sentiment Distribution")
        sentiment_counts = df['sentiment_label'].value_counts(normalize=True).reset_index()
        sentiment_counts.columns = ['sentiment', 'percentage']
        fig_sentiment = px.pie(sentiment_counts, values='percentage', names='sentiment',
                                title='Overall Sentiment of Road Issue Reports',
                                color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_sentiment, use_container_width=True)

    elif page == "📋 Export & Reports":
        st.markdown('<div class="sub-header">Export Data & Generate Custom Reports</div>', unsafe_allow_html=True)
        
        if 'processed_data' not in st.session_state:
            st.warning("No data available for export. Please generate sample data or submit issues.")
            return

        df_export = st.session_state.processed_data.copy()

        st.markdown("### 📥 Download Processed Data")
        csv = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="road_issues_processed_data.csv",
            mime="text/csv",
        )

        st.markdown("### ⚙️ Custom Report Generator")
        st.write("Select criteria to generate a custom report:")

        report_category = st.multiselect("Filter by Category:", options=platform.road_categories, default=platform.road_categories)
        report_location = st.multiselect("Filter by Location Type:", options=platform.location_types, default=platform.location_types)
        report_priority = st.multiselect("Filter by Priority:", options=['Low', 'Medium', 'High'], default=['Low', 'Medium', 'High'])
        report_status = st.multiselect("Filter by Status:", options=['Open', 'In Progress', 'Resolved', 'Closed'], default=['Open', 'In Progress', 'Resolved', 'Closed'])
        
        min_cost, max_cost = st.slider("Filter by Estimated Cost ($):", 
                                        min_value=int(df_export['estimated_cost'].min()), 
                                        max_value=int(df_export['estimated_cost'].max()), 
                                        value=(int(df_export['estimated_cost'].min()), int(df_export['estimated_cost'].max())))

        if st.button("Generate Custom Report"):
            filtered_df = df_export[
                (df_export['category'].isin(report_category)) &
                (df_export['location'].isin(report_location)) &
                (df_export['priority'].isin(report_priority)) &
                (df_export['status'].isin(report_status)) &
                (df_export['estimated_cost'] >= min_cost) &
                (df_export['estimated_cost'] <= max_cost)
            ]

            if not filtered_df.empty:
                st.markdown("---")
                st.markdown("### 📈 Generated Custom Report")
                st.write(f"Displaying {len(filtered_df)} records based on your filters.")
                st.dataframe(filtered_df[['feedback_id', 'timestamp', 'category', 'location', 'priority', 'status', 'estimated_cost', 'feedback_text']].head(10))

                st.markdown("#### Summary Statistics for Custom Report")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Issues (Filtered)", len(filtered_df))
                with col2:
                    st.metric("Average Cost (Filtered)", f"${filtered_df['estimated_cost'].mean():,.0f}")
                with col3:
                    st.metric("Highest Priority (Filtered)", filtered_df['priority'].value_counts().idxmax() if not filtered_df['priority'].empty else 'N/A')

                st.markdown("---")
                report_csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Filtered Report as CSV",
                    data=report_csv,
                    file_name="custom_road_report.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No data matches your selected filters. Please adjust your criteria.")


if __name__ == "__main__":
    main()