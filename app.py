import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import base64
from sklearn.feature_selection import SelectFromModel

# --- Load dataset ---
df = pd.read_csv("mushrooms.csv")

cap_shape_dict = {
    'b' : 'Bell',
    'c' : 'Conical',
    'x' : 'Convex',
    'f' : 'Flat',
    'k' : 'Knobbed',
    's' : 'Sunken'
}

cap_surface_dict = {
    'f' : 'Fibrous',
    'g' : 'Grooves',
    'y' : 'Scaly',
    's' : 'Smooth'
}

cap_color_dict = {
    'n' : 'Brown',
    'b' : 'Buff',
    'c' : 'Cinnamon',
    'g' : 'Gray',
    'r' : 'Green',
    'p' : 'Pink',
    'u' : 'Purple',
    'e' : 'Red',
    'w' : 'White',
    'y' : 'Yellow'
}

bruises_dict = {
    't' : 'Bruises',
    'f' : 'None'
}

odor_dict = {
    'a' : 'Almond',
    'l' : 'Anise',
    'c' : 'Creosote',
    'y' : 'Fishy',
    'f' : 'Foul',
    'm' : 'Musty',
    'n' : 'None',
    'p' : 'Pungent',
    's' : 'Spicy'
}

gill_attachment_dict = {
    'a' : 'Attached',
    'd' : 'Descending',
    'f' : 'Free',
    'n' : 'Notched'
}

gill_spacing_dict = {
    'c' : 'Close',
    'w' : 'Crowded',
    'd' : 'Distant'
}

gill_size_dict = {
    'b' : 'Broad',
    'n' : 'Narrow'
}

gill_color_dict = {
    'k' : 'Black',
    'n' : 'Brown',
    'b' : 'Buff',
    'h' : 'Chocolate',
    'g' : 'Gray',
    'r' : 'Green',
    'o' : 'Orange',
    'p' : 'Pink',
    'u' : 'Purple',
    'e' : 'Red',
    'w' : 'White',
    'y' : 'Yellow'
}

stalk_shape_dict = {
    'e' : 'Enlarging',
    't' : 'Tapering'
}

stalk_root_dict = {
    'b' : 'Bulbous',
    'c' : 'Club',
    'u' : 'Cup',
    'e' : 'Equal',
    'z' : 'Rhizomorphs',
    'r' : 'Rooted',
    '?' : 'None'
}

stalk_surface_above_ring_dict = {
    'f' : 'Fibrous',
    'y' : 'Scaly',
    'k' : 'Silky',
    's' : 'Smooth'
}

stalk_surface_below_ring_dict = {
    'f' : 'Fibrous',
    'y' : 'Scaly',
    'k' : 'Silky',
    's' : 'Smooth'
}

stalk_color_above_ring_dict = {
    'n' : 'Brown',
    'b' : 'Buff',
    'c' : 'Cinnamon',
    'g' : 'Gray',
    'o' : 'Orange',
    'p' : 'Pink',
    'e' : 'Red',
    'w' : 'White',
    'y' : 'Yellow'
}

stalk_color_below_ring_dict = {
    'n' : 'Brown',
    'b' : 'Buff',
    'c' : 'Cinnamon',
    'g' : 'Gray',
    'o' : 'Orange',
    'p' : 'Pink',
    'e' : 'Red',
    'w' : 'White',
    'y' : 'Yellow'
}

veil_type_dict = {
    'p' : 'Partial',
    'u' : 'Universal'
}

veil_color_dict = {
    'n' : 'Brown',
    'o' : 'Orange',
    'w' : 'White',
    'y' : 'Yellow'
}

ring_number_dict = {
    'n' : 'None',
    'o' : 'One',
    't' : 'Two'
}

ring_type_dict = {
    'c' : 'Cobwebby',
    'e' : 'Evanescent',
    'f' : 'Flaring',
    'l' : 'Large',
    'n' : 'None',
    'p' : 'Pendant',
    's' : 'Sheathing',
    'z' : 'Zone'
}

spore_print_color_dict = {
    'k' : 'Black',
    'n' : 'Brown',
    'b' : 'Buff',
    'h' : 'Chocolate',
    'r' : 'Green',
    'o' : 'Orange',
    'u' : 'Purple',
    'w' : 'White',
    'y' : 'Yellow'
}

population_dict = {
    'a' : 'Abundant',
    'c' : 'Clustered',
    'n' : 'Numerous',
    's' : 'Scattered',
    'v' : 'Several',
    'y' : 'Solitary'
}

habitat_dict = {
    'g' : 'Grasses',
    'l' : 'Leaves',
    'm' : 'Meadows',
    'p' : 'Paths',
    'u' : 'Urban',
    'w' : 'Waste',
    'd' : 'Woods'
}

attribute_mappings = {
    'cap-shape' : cap_shape_dict,
    'cap-surface' : cap_surface_dict,
    'cap-color' : cap_color_dict,
    'bruises' : bruises_dict,
    'odor' : odor_dict,
    'gill-attachment' : gill_attachment_dict,
    'gill-spacing' : gill_spacing_dict,
    'gill-size' : gill_size_dict,
    'gill-color' : gill_color_dict,
    'stalk-shape' : stalk_shape_dict,
    'stalk-root' : stalk_root_dict,
    'stalk-surface-above-ring' : stalk_surface_above_ring_dict,
    'stalk-surface-below-ring' : stalk_surface_below_ring_dict,
    'stalk-color-above-ring' : stalk_color_above_ring_dict,
    'stalk-color-below-ring' : stalk_color_below_ring_dict,
    'veil-type' : veil_type_dict,
    'veil-color' : veil_color_dict,
    'ring-number' : ring_number_dict,
    'ring-type' : ring_type_dict,
    'spore-print-color' : spore_print_color_dict,
    'population' : population_dict,
    'habitat' : habitat_dict

}

# --- Encode categorical variables ---
label_encoders = {}
df_encoded = df.copy()
for col in df.columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

X = df_encoded.drop("class", axis=1)
y = df_encoded["class"]

fs_model = DecisionTreeClassifier(random_state=42)
fs_model.fit(X, y)

selector = SelectFromModel(fs_model, threshold="median", prefit=True)
x = selector.transform

selected_columns = df_encoded.drop("class", axis=1).columns[selector.get_support()]

# --- Train model ---
X_selected = selector.transform(X)
model = DecisionTreeClassifier()
model.fit(X_selected, y)

# --- Encode local image to base64 ---
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# 🖼️ Background image setup (place the image in same folder and name it 'mushroom_bg.jpg')
img_base64 = get_img_as_base64("Screenshot 2025-07-09 130611.png")

# --- Inject CSS (dark theme + Great Vibes font + UI styling) ---
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Great+Vibes&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Great Vibes', cursive;
            background-color: #1e1e1e;
            color: #ffffff;
        }}

        .stApp {{
            background-image: url("data:image/jpg;base64,{img_base64}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        .block-container {{
            background-color: rgba(0, 0, 0, 0.6);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 0 20px rgba(0,0,0,0.9);
        }}

        h1, h2, h3, h4 {{
            color: #ff99cc;
            text-shadow: 2px 2px 4px #000000aa;
        }}

        .stButton>button {{
            background-color: #ff99cc;
            color: black;
            font-size: 1.2rem;
            border-radius: 10px;
            padding: 10px 30px;
            border: none;
            box-shadow: 2px 2px 6px #00000099;
        }}

        .stSelectbox>div>div {{
            font-size: 1.1rem;
            color: black;
        }}

        .stAlert {{
            font-size: 1.5rem;
        }}
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.title("🍄 MUSHROOM ID 🍄")

st.markdown("""
    ## 🍃 Welcome to the Enchanted Forest 🍃  
    🌸 Use this magical tool to check if your mushroom is **safe** 🟢 or **dangerous** 🔴.  
    🧠 Simply choose its characteristics and let the forest whisper the truth...
""")

# --- User Inputs ---
user_input = {}
for col in selected_columns:
    if col in attribute_mappings:
        readable_options = [attribute_mappings[col][code] for code in df[col].unique()]
        selected_readable = st.selectbox(col, readable_options)

        code = [k for k, v in attribute_mappings[col].items() if v == selected_readable][0]
        user_input[col] = code
    else:
        options = df[col].unique()
        user_input[col] = st.selectbox(col, options)
    

# --- Prediction ---
if st.button("🔍 Reveal Mushroom Fate"):
    # Start with user inputs
    input_data = user_input.copy()

    # Add placeholder values for missing features (not shown to the user)
    full_feature_list = df.drop("class", axis=1).columns
    for col in full_feature_list:
        if col not in input_data:
            # Use the mode (most common value) as a safe placeholder
            input_data[col] = df[col].mode()[0]

    # Create input dataframe
    input_df = pd.DataFrame([input_data])

    # Encode using fitted LabelEncoders
    for col in input_df.columns:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Now safe to apply feature selector
    input_df = selector.transform(input_df)

    # Predict
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("🌟 This mushroom is **EDIBLE**! You may feast safely 🍽️")
    else:
        st.error("☠️ This mushroom is **POISONOUS**! Do **NOT** consume it!")
