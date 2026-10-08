from pathlib import Path

from flask import Flask, render_template, request, session
import pandas as pd
import joblib


# ============================================================
# APP CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "risk_model.pkl"

app = Flask(__name__)

# Secret key required for Flask sessions
app.secret_key = "coastal-resilience-demo-key"


# ============================================================
# LOAD AI MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"AI model not found at: {MODEL_PATH}\n"
        "Please run train_model.py first."
    )

risk_model = joblib.load(MODEL_PATH)

print("AI risk model loaded successfully.")
print(f"Model path: {MODEL_PATH}")


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# COMMUNITY DASHBOARD
# ============================================================

@app.route("/community")
def community():

    return render_template(
        "dashboard/community.html"
    )


# ============================================================
# AUTHORITY DASHBOARD
# ============================================================

@app.route("/authority")
def authority():

    return render_template(
        "dashboard/authority.html"
    )


# ============================================================
# GIS RISK MAP
# ============================================================

@app.route("/map")
def map_view():

    return render_template(
        "map.html"
    )


# ============================================================
# DISASTER SCENARIO
# ============================================================

@app.route("/scenario", methods=["GET", "POST"])
def scenario():

    # --------------------------------------------------------
    # OPEN SCENARIO PAGE
    # --------------------------------------------------------

    if request.method == "GET":

        return render_template(
            "scenario.html",
            scenario=session.get("scenario")
        )


    # --------------------------------------------------------
    # GET FORM VALUES
    # --------------------------------------------------------

    disaster_type = request.form.get(
        "disaster_type",
        ""
    ).strip()

    try:

        rainfall = float(
            request.form.get("rainfall", 0)
        )

        wind_speed = float(
            request.form.get("wind_speed", 0)
        )

        tide_level = float(
            request.form.get("tide_level", 0)
        )

        elevation = float(
            request.form.get("elevation", 0)
        )

        population = float(
            request.form.get("population", 0)
        )

    except (TypeError, ValueError):

        return (
            "Invalid scenario input. "
            "Please enter valid numerical values.",
            400
        )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not disaster_type:

        return (
            "Please select a disaster type.",
            400
        )


    if rainfall < 0:

        return (
            "Rainfall cannot be negative.",
            400
        )


    if wind_speed < 0:

        return (
            "Wind speed cannot be negative.",
            400
        )


    if tide_level < 0:

        return (
            "Tide level cannot be negative.",
            400
        )


    if elevation < 0:

        return (
            "Elevation cannot be negative.",
            400
        )


    if population < 0:

        return (
            "Population cannot be negative.",
            400
        )


    # --------------------------------------------------------
    # DISASTER DISPLAY NAMES
    # --------------------------------------------------------

    disaster_names = {

        "cyclone":
            "Cyclone",

        "storm_surge":
            "Storm Surge",

        "coastal_flood":
            "Coastal Flood",

        "heavy_rainfall":
            "Heavy Rainfall"

    }


    display_disaster_type = disaster_names.get(
        disaster_type,
        disaster_type
    )


    # ========================================================
    # PREPARE AI MODEL INPUT
    # ========================================================

    input_data = pd.DataFrame(
        [
            {
                "rainfall": rainfall,
                "wind_speed": wind_speed,
                "tide_level": tide_level,
                "elevation": elevation,
                "population": population
            }
        ]
    )


    # ========================================================
    # AI PREDICTION
    # ========================================================

    prediction = risk_model.predict(
        input_data
    )

    predicted_class = int(
        prediction[0]
    )


    # --------------------------------------------------------
    # RISK LABEL
    # --------------------------------------------------------

    risk_labels = {

        0: "Low",

        1: "Medium",

        2: "High"

    }


    predicted_risk = risk_labels.get(
        predicted_class,
        "Unknown"
    )


    # ========================================================
    # RISK SCORE
    # ========================================================

    rainfall_score = min(
        (rainfall / 200) * 25,
        25
    )


    wind_score = min(
        (wind_speed / 100) * 25,
        25
    )


    tide_score = min(
        (tide_level / 2.0) * 20,
        20
    )


    elevation_score = max(
        15 - (elevation / 30) * 15,
        0
    )


    population_score = min(
        (population / 5000) * 15,
        15
    )


    risk_score = (

        rainfall_score

        + wind_score

        + tide_score

        + elevation_score

        + population_score

    )


    risk_score = round(
        min(
            max(risk_score, 0),
            100
        )
    )


    # ========================================================
    # SAVE SCENARIO
    # ========================================================

    scenario_data = {

        "disaster_type":
            display_disaster_type,

        "rainfall":
            rainfall,

        "wind_speed":
            wind_speed,

        "tide_level":
            tide_level,

        "elevation":
            elevation,

        "population":
            population,

        "predicted_risk":
            predicted_risk,

        "risk_score":
            risk_score

    }


    session["scenario"] = scenario_data


    # --------------------------------------------------------
    # CLEAR PREVIOUS ALERT
    # --------------------------------------------------------

    session.pop(
        "alert_simulation",
        None
    )


    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    print()
    print("==============================")
    print("AI DISASTER RISK PREDICTION")
    print("==============================")
    print(
        "Disaster Type:",
        display_disaster_type
    )
    print(
        "Rainfall:",
        rainfall,
        "mm"
    )
    print(
        "Wind Speed:",
        wind_speed,
        "km/h"
    )
    print(
        "Tide Level:",
        tide_level,
        "m"
    )
    print(
        "Elevation:",
        elevation,
        "m"
    )
    print(
        "Population:",
        population
    )
    print(
        "Predicted Risk:",
        predicted_risk
    )
    print(
        "Risk Score:",
        risk_score,
        "/ 100"
    )
    print("==============================")
    print()


    # ========================================================
    # SHOW RESULT
    # ========================================================

    return render_template(
        "scenario.html",
        scenario=scenario_data
    )


# ============================================================
# ALERT SIMULATION
# ============================================================

@app.route("/alerts", methods=["GET", "POST"])
def alerts():

    # --------------------------------------------------------
    # GET LATEST SCENARIO
    # --------------------------------------------------------

    scenario_data = session.get(
        "scenario",
        {}
    )


    # --------------------------------------------------------
    # GET PREVIOUS ALERT
    # --------------------------------------------------------

    alert_result = session.get(
        "alert_simulation",
        None
    )


    # ========================================================
    # ALERT FORM SUBMISSION
    # ========================================================

    if request.method == "POST":

        # ----------------------------------------------------
        # CHECK WHETHER A SCENARIO EXISTS
        # ----------------------------------------------------

        if not scenario_data:

            return render_template(
                "alerts.html",
                scenario={},
                alert_result=None,
                alert_error=(
                    "Please run a disaster scenario "
                    "before generating an alert."
                )
            )


        # ----------------------------------------------------
        # FORM VALUES
        # ----------------------------------------------------

        channel = request.form.get(
            "channel",
            "Web"
        )


        village = request.form.get(
            "village",
            "Demo Coastal Village"
        ).strip()


        language = request.form.get(
            "language",
            "English"
        )


        if not village:

            village = "Demo Coastal Village"


        # ----------------------------------------------------
        # SCENARIO VALUES
        # ----------------------------------------------------

        disaster_type = scenario_data.get(
            "disaster_type",
            "Coastal Hazard"
        )


        risk_level = scenario_data.get(
            "predicted_risk",
            "Unknown"
        )


        risk_score = scenario_data.get(
            "risk_score",
            0
        )


        rainfall = scenario_data.get(
            "rainfall",
            0
        )


        wind_speed = scenario_data.get(
            "wind_speed",
            0
        )


        tide_level = scenario_data.get(
            "tide_level",
            0
        )


        # ====================================================
        # SAFETY ACTION
        # ====================================================

        if risk_level == "High":

            english_action = (
                "Move to a safe location or "
                "designated shelter immediately."
            )

            tamil_action = (
                "உடனடியாக பாதுகாப்பான இடம் அல்லது "
                "நியமிக்கப்பட்ட நிவாரண முகாமிற்கு செல்லவும்."
            )

            hindi_action = (
                "तुरंत सुरक्षित स्थान या "
                "निर्धारित आश्रय स्थल पर जाएं।"
            )


        elif risk_level == "Medium":

            english_action = (
                "Stay alert, monitor official "
                "updates and prepare for evacuation."
            )

            tamil_action = (
                "விழிப்புடன் இருந்து அதிகாரப்பூர்வ "
                "தகவல்களை கண்காணித்து வெளியேற்றத்திற்கு தயாராகுங்கள்."
            )

            hindi_action = (
                "सतर्क रहें, आधिकारिक जानकारी देखें "
                "और निकासी के लिए तैयार रहें।"
            )


        else:

            english_action = (
                "Continue monitoring official "
                "weather and safety updates."
            )

            tamil_action = (
                "அதிகாரப்பூர்வ வானிலை மற்றும் "
                "பாதுகாப்பு தகவல்களை தொடர்ந்து கண்காணிக்கவும்."
            )

            hindi_action = (
                "आधिकारिक मौसम और सुरक्षा "
                "जानकारी की निगरानी जारी रखें।"
            )


        # ====================================================
        # CREATE MESSAGE
        # ====================================================

        if language == "Tamil":

            message = (
                f"எச்சரிக்கை: {disaster_type} - "
                f"{village}. ஆபத்து நிலை: "
                f"{risk_level} ({risk_score}/100). "
                f"மழை: {rainfall} mm, "
                f"காற்று: {wind_speed} km/h, "
                f"அலை மட்டம்: {tide_level} m. "
                f"{tamil_action}"
            )


        elif language == "Hindi":

            message = (
                f"चेतावनी: {disaster_type} - "
                f"{village}. जोखिम स्तर: "
                f"{risk_level} ({risk_score}/100). "
                f"वर्षा: {rainfall} mm, "
                f"हवा: {wind_speed} km/h, "
                f"ज्वार स्तर: {tide_level} m. "
                f"{hindi_action}"
            )


        else:

            message = (
                f"ALERT: {disaster_type} detected "
                f"for {village}. "
                f"Risk Level: {risk_level} "
                f"({risk_score}/100). "
                f"Rainfall: {rainfall} mm, "
                f"Wind: {wind_speed} km/h, "
                f"Tide: {tide_level} m. "
                f"{english_action}"
            )


        # ====================================================
        # SAVE ALERT SIMULATION
        # ====================================================

        alert_result = {

            "channel":
                channel,

            "village":
                village,

            "language":
                language,

            "risk_level":
                risk_level,

            "risk_score":
                risk_score,

            "message":
                message

        }


        session["alert_simulation"] = (
            alert_result
        )


    # ========================================================
    # DISPLAY ALERT PAGE
    # ========================================================

    return render_template(
        "alerts.html",
        scenario=scenario_data,
        alert_result=alert_result
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )