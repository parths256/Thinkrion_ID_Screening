from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import os
import re
import base64
import easyocr
import cv2
import numpy as np

from deepface import DeepFace


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

reader = easyocr.Reader(["en"], gpu=False)

LAST_DOCUMENT_PATH = None
LAST_FIELDS = None
LAST_TAMPERING = None


# =========================================================
# OCR FIELD EXTRACTION
# =========================================================

def extract_fields(text):

    fields = {
        "document_type": "Unknown",
        "name": "Not detected",
        "dob": "Not detected",
        "gender": "Not detected",
        "id_number": "Not detected",
        "vid": "Not detected",
        "issuer": "Not detected",
        "validation": "Not checked",
        "risk_score": 0,
        "risk_level": "LOW RISK",
        "reasons": []
    }

    lower_text = text.lower()

    # DOCUMENT TYPE
    if "aadhaar" in lower_text or "uidai" in lower_text:

        fields["document_type"] = "Aadhaar Card"
        fields["issuer"] = "UIDAI - Government of India"

    # GENDER
    gender_match = re.search(
        r"\b(MALE|FEMALE|TRANSGENDER)\b",
        text,
        re.IGNORECASE
    )

    if gender_match:
        fields["gender"] = gender_match.group(1).upper()

    # DOB
    dob_patterns = [
        r"DOB[:\s\-]*([0-3]?\d)[/\-\s]+([0-1]?\d)[/\-\s]+(\d{4})",
        r"Date\s*of\s*Birth[:\s\-]*([0-3]?\d)[/\-\s]+([0-1]?\d)[/\-\s]+(\d{4})",
        r"\b([0-3]?\d)[/\-]([0-1]?\d)[/\-](\d{4})\b"
    ]

    for pattern in dob_patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            day = match.group(1).zfill(2)
            month = match.group(2).zfill(2)
            year = match.group(3)

            fields["dob"] = f"{day}/{month}/{year}"
            break

    # AADHAAR NUMBER
    aadhaar_matches = re.findall(
        r"\b\d{4}\s+\d{4}\s+\d{4}\b",
        text
    )

    if aadhaar_matches:

        aadhaar = aadhaar_matches[0]

        digits = aadhaar.replace(
            " ",
            ""
        )

        fields["id_number"] = (
            f"XXXX XXXX {digits[-4:]}"
        )

    # VID
    vid_match = re.search(
        r"VID\s*[:\-]?\s*(\d{4})\s*(\d{4})\s*(\d{4})\s*(\d{4})",
        text,
        re.IGNORECASE
    )

    if vid_match:

        fields["vid"] = (
            "XXXX XXXX XXXX "
            + vid_match.group(4)
        )

    # NAME DETECTION
    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    blocked_words = [
        "government",
        "india",
        "unique",
        "identification",
        "authority",
        "aadhaar",
        "information",
        "address",
        "male",
        "female",
        "dob",
        "vid",
        "enrolment",
        "proof",
        "citizenship",
        "mobile",
        "download",
        "authentication",
        "document",
        "your",
        "uidai"
    ]

    anchor_index = None

    for i, line in enumerate(lines):

        low = line.lower()

        if (
            "dob" in low
            or "male" in low
            or "female" in low
        ):

            anchor_index = i
            break

    candidates = []

    if anchor_index is not None:

        start = max(
            0,
            anchor_index - 8
        )

        nearby_lines = lines[
            start:anchor_index
        ]

        for line in nearby_lines:

            cleaned = re.sub(
                r"[^A-Za-z ]",
                "",
                line
            )

            cleaned = re.sub(
                r"\s+",
                " ",
                cleaned
            ).strip()

            words = cleaned.split()

            if not (
                2 <= len(words) <= 3
            ):
                continue

            if any(
                blocked in cleaned.lower()
                for blocked in blocked_words
            ):
                continue

            if len(cleaned) > 30:
                continue

            if any(
                len(word) < 2
                for word in words
            ):
                continue

            score = 0

            score += sum(
                2
                for word in words
                if word[0].isupper()
            )

            if len(words) == 2:
                score += 5

            candidates.append(
                (
                    score,
                    cleaned
                )
            )

    if candidates:

        candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        fields["name"] = (
            candidates[0][1]
        )

    # OCR RISK SCORE
    risk = 0
    reasons = []

    if fields["document_type"] == "Unknown":

        risk += 20
        reasons.append(
            "Document type not identified"
        )

    if fields["id_number"] == "Not detected":

        risk += 25
        reasons.append(
            "ID number not detected"
        )

    if fields["dob"] == "Not detected":

        risk += 15
        reasons.append(
            "DOB not detected"
        )

    if fields["name"] == "Not detected":

        risk += 15
        reasons.append(
            "Name not detected"
        )

    if fields["gender"] == "Not detected":

        risk += 10
        reasons.append(
            "Gender not detected"
        )

    fields["risk_score"] = min(
        risk,
        100
    )

    if risk <= 20:

        fields["risk_level"] = "LOW RISK"

        fields["validation"] = (
            "Document appears structurally valid"
        )

    elif risk <= 50:

        fields["risk_level"] = (
            "MANUAL REVIEW"
        )

        fields["validation"] = (
            "Some fields could not be confidently verified"
        )

    else:

        fields["risk_level"] = (
            "HIGH RISK"
        )

        fields["validation"] = (
            "Multiple verification issues detected"
        )

    fields["reasons"] = reasons

    return fields


# =========================================================
# TAMPERING ANALYSIS
# =========================================================

def analyze_tampering(image_path):

    result = {
        "score": 0,
        "status": "LOW SUSPICION",
        "blur_score": 0,
        "edge_density": 0,
        "brightness_std": 0,
        "reasons": []
    }

    image = cv2.imread(
        image_path
    )

    if image is None:

        result["score"] = 50
        result["status"] = "MANUAL REVIEW"

        result["reasons"].append(
            "Image could not be analyzed"
        )

        return result

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    tamper_score = 0

    # SHARPNESS
    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    blur_score = laplacian.var()

    result["blur_score"] = round(
        float(blur_score),
        2
    )

    if blur_score < 35:

        tamper_score += 20

        result["reasons"].append(
            "Image appears unusually blurred"
        )

    # EDGE ANALYSIS
    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_pixels = np.count_nonzero(
        edges
    )

    total_pixels = (
        edges.shape[0]
        *
        edges.shape[1]
    )

    edge_density = (
        edge_pixels
        /
        total_pixels
    )

    result["edge_density"] = round(
        float(edge_density),
        4
    )

    if edge_density > 0.25:

        tamper_score += 25

        result["reasons"].append(
            "Unusually high edge density detected"
        )

    elif edge_density < 0.01:

        tamper_score += 15

        result["reasons"].append(
            "Very low structural detail detected"
        )

    # BRIGHTNESS
    brightness_std = np.std(
        gray
    )

    result["brightness_std"] = round(
        float(brightness_std),
        2
    )

    if brightness_std > 80:

        tamper_score += 15

        result["reasons"].append(
            "Strong brightness inconsistencies detected"
        )

    # LOCAL TEXTURE ANALYSIS
    h, w = gray.shape

    block_h = max(
        h // 4,
        1
    )

    block_w = max(
        w // 4,
        1
    )

    block_variances = []

    for y in range(
        0,
        h,
        block_h
    ):

        for x in range(
            0,
            w,
            block_w
        ):

            block = gray[
                y:min(y + block_h, h),
                x:min(x + block_w, w)
            ]

            if block.size > 0:

                block_variances.append(
                    float(
                        np.var(block)
                    )
                )

    if block_variances:

        variance_range = (
            max(block_variances)
            -
            min(block_variances)
        )

        if variance_range > 5000:

            tamper_score += 20

            result["reasons"].append(
                "Large local texture differences detected"
            )

    tamper_score = min(
        tamper_score,
        100
    )

    result["score"] = tamper_score

    if tamper_score <= 20:

        result["status"] = (
            "LOW SUSPICION"
        )

        if not result["reasons"]:

            result["reasons"].append(
                "No major image anomalies detected"
            )

    elif tamper_score <= 50:

        result["status"] = (
            "MANUAL REVIEW"
        )

    else:

        result["status"] = (
            "HIGH SUSPICION"
        )

        result["reasons"].append(
            "Document should be manually reviewed"
        )

    return result


# =========================================================
# FINAL RISK CALCULATION
# =========================================================

def calculate_final_risk(
    ocr_score,
    tamper_score,
    face_verified
):

    # Face risk:
    # match = 0 risk
    # no match = 100 risk

    face_risk = (
        0
        if face_verified
        else 100
    )

    final_score = (
        (ocr_score * 0.35)
        +
        (tamper_score * 0.35)
        +
        (face_risk * 0.30)
    )

    final_score = round(
        final_score
    )

    if final_score <= 25:

        level = "LOW RISK"

    elif final_score <= 55:

        level = "MANUAL REVIEW"

    else:

        level = "HIGH RISK"

    return (
        final_score,
        level,
        face_risk
    )


# =========================================================
# HOME
# =========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    global LAST_DOCUMENT_PATH
    global LAST_FIELDS
    global LAST_TAMPERING

    extracted_text = None
    fields = None
    filename = None
    tampering = None

    if request.method == "POST":

        file = request.files.get(
            "document"
        )

        if file and file.filename:

            filename = secure_filename(
                file.filename
            )

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(
                filepath
            )

            LAST_DOCUMENT_PATH = filepath

            # OCR
            results = reader.readtext(
                filepath
            )

            detected_lines = [
                result[1]
                for result in results
            ]

            extracted_text = "\n".join(
                detected_lines
            )

            # OCR / FIELD CHECK
            fields = extract_fields(
                extracted_text
            )

            # TAMPERING CHECK
            tampering = analyze_tampering(
                filepath
            )

            LAST_FIELDS = fields
            LAST_TAMPERING = tampering

    return render_template(
        "index.html",
        extracted_text=extracted_text,
        fields=fields,
        filename=filename,
        tampering=tampering
    )


# =========================================================
# FACE VERIFICATION
# =========================================================

@app.route(
    "/verify-face",
    methods=["POST"]
)
def verify_face():

    global LAST_DOCUMENT_PATH
    global LAST_FIELDS
    global LAST_TAMPERING

    if not LAST_DOCUMENT_PATH:

        return jsonify({
            "success": False,
            "message":
                "Please upload a document first."
        })

    data = request.get_json()

    if not data or "image" not in data:

        return jsonify({
            "success": False,
            "message":
                "No webcam image received."
        })

    try:

        image_data = data[
            "image"
        ]

        if "," in image_data:

            image_data = (
                image_data.split(",")[1]
            )

        image_bytes = (
            base64.b64decode(
                image_data
            )
        )

        live_face_path = os.path.join(
            UPLOAD_FOLDER,
            "live_face.jpg"
        )

        with open(
            live_face_path,
            "wb"
        ) as f:

            f.write(
                image_bytes
            )

        # FACE CHECK
        result = DeepFace.verify(

            img1_path=
                LAST_DOCUMENT_PATH,

            img2_path=
                live_face_path,

            model_name=
                "Facenet512",

            detector_backend=
                "retinaface",

            enforce_detection=
                False,

            align=
                True
        )

        verified = bool(
            result.get(
                "verified",
                False
            )
        )

        distance = float(
            result.get(
                "distance",
                1
            )
        )

        threshold = float(
            result.get(
                "threshold",
                1
            )
        )

        if threshold > 0:

            similarity = (
                1
                -
                (
                    distance
                    /
                    (threshold * 2)
                )
            ) * 100

        else:

            similarity = 0

        similarity = max(
            0,
            min(
                100,
                similarity
            )
        )

        # FINAL RISK SCORE
        ocr_score = (
            LAST_FIELDS["risk_score"]
            if LAST_FIELDS
            else 50
        )

        tamper_score = (
            LAST_TAMPERING["score"]
            if LAST_TAMPERING
            else 50
        )

        final_score, final_level, face_risk = (
            calculate_final_risk(
                ocr_score,
                tamper_score,
                verified
            )
        )

        reasons = []

        if ocr_score > 20:

            reasons.append(
                "OCR or identity field issues detected"
            )

        if tamper_score > 20:

            reasons.append(
                "Document integrity anomalies detected"
            )

        if verified:

            reasons.append(
                "Live face matched document identity"
            )

        else:

            reasons.append(
                "Live face did not match document identity"
            )

        if final_level != "LOW RISK":

            reasons.append(
                "Officer review recommended"
            )

        return jsonify({

            "success": True,

            "verified": verified,

            "similarity": round(
                similarity,
                1
            ),

            "ocr_score":
                ocr_score,

            "tamper_score":
                tamper_score,

            "face_risk":
                face_risk,

            "final_score":
                final_score,

            "final_level":
                final_level,

            "reasons":
                reasons
        })

    except Exception as e:

        print(
            "FACE ERROR:",
            e
        )

        return jsonify({

            "success": False,

            "message":
                "Face verification error: "
                + str(e)
        })


if __name__ == "__main__":

    app.run(
        debug=True
    )