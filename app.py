#app.py

"""
NeuroScan AI — Flask Backend (Optimized + Gemini AI Report)
Run: python app.py  →  Open: http://localhost:5000

Install dependencies:
  pip install flask flask-cors tensorflow pillow numpy opencv-python google-generativeai
"""

import os, io, sys, base64, traceback, time, json
import concurrent.futures
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ══════════════════════════════════════════════════════════════
# CONFIG — edit these
# ══════════════════════════════════════════════════════════════
MODEL_PATH        = 'best_densenet_model.h5'
IMG_SIZE          = (224, 224)
CLASS_NAMES       = [
    'MildImpairment',
    'ModerateImpairment',
    'NoImpairment',
    'VeryMildImpairment'
]
NO_IMPAIRMENT_IDX = 2

# ── Paste your Google AI Studio API key here ──────────────────
GEMINI_API_KEY    = ""
# Model options (try in order if one doesn't work):
#   'gemini-2.0-flash'                      ← most reliable, widely available
#   'gemini-2.5-flash-lite-preview-06-17'   ← fastest/cheapest, may need allowlist
#   'gemini-1.5-flash'                      ← fallback if above fail
GEMINI_MODEL = "gemini-2.5-flash-lite"
# ══════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET","POST","OPTIONS"],
                              "allow_headers": ["Content-Type"]}})

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.route('/predict', methods=['OPTIONS'])
@app.route('/health',  methods=['OPTIONS'])
@app.route('/debug',   methods=['OPTIONS'])
def preflight():
    r = jsonify({'ok': True})
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return r, 200


# ── Load model ────────────────────────────────────────────────
print("\n" + "="*55)
print("  NeuroScan AI — Starting")
print("="*55)

if not os.path.exists(MODEL_PATH):
    print(f"\n  ERROR: '{MODEL_PATH}' not found in {os.path.abspath('.')}")
    sys.exit(1)

print(f"\n  Loading model: {MODEL_PATH} ...")
t0 = time.time()
model = load_model(MODEL_PATH)
print(f"  Loaded in {time.time()-t0:.1f}s — input: {model.input_shape}")

# ── Auto-detect last Conv2D layer ─────────────────────────────
last_conv = 'conv5_block32_concat'
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv = layer.name
        break
if last_conv is None:
    for name in ['conv5_block16_2_conv','conv5_block32_2_conv','top_conv','block5_conv3']:
        try: model.get_layer(name); last_conv = name; break
        except: continue
if last_conv is None:
    last_conv = model.layers[-3].name
print(f"  Grad-CAM layer : '{last_conv}'")

# ── Combined model (single forward pass) ─────────────────────
grad_model = tf.keras.Model(
    inputs  = model.inputs,
    outputs = [model.get_layer(last_conv).output, model.output]
)

# ── Warmup ────────────────────────────────────────────────────
print("  Warming up model...")
t0 = time.time()
dummy = np.zeros((1, *IMG_SIZE, 3), dtype=np.float32)
_ = grad_model(dummy, training=False)
_ = grad_model(dummy, training=False)
print(f"  Warmup done in {time.time()-t0:.1f}s")

# ── Persistent thread pool (avoid creating per-request) ───────
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)


# ── Gemini system prompt ──────────────────────────────────────
SYSTEM_PROMPT = """
You are a specialist clinical AI assistant for Alzheimer's disease and 
neurodegenerative conditions. You explain MRI scan results in warm, 
compassionate, plain English that a 60-year-old patient and their 
family can fully understand without any medical background.

You MUST respond with ONLY a valid JSON object — no text before or 
after it, no markdown code fences, no explanation outside the JSON.

The JSON must follow this EXACT schema every time:

{
  "what_this_means": "2-3 sentences in plain English explaining what 
    this specific stage means for this specific patient. No jargon. 
    Start with 'Your scan...'",

  "possible_reasons": [
    "Reason 1 as a complete sentence starting with a possible factor",
    "Reason 2...",
    "Reason 3...",
    "Reason 4..."
  ],

  "symptoms_to_watch": [
    "Symptom 1 as a clear, specific, observable description",
    "Symptom 2...",
    "Symptom 3...",
    "Symptom 4..."
  ],

  "diet_recommendations": [
    "Specific dietary advice 1 with brief reason why it helps",
    "Specific dietary advice 2...",
    "Specific dietary advice 3...",
    "Specific dietary advice 4..."
  ],

  "lifestyle_practices": [
    "Specific practice 1 with brief reason",
    "Specific practice 2...",
    "Specific practice 3...",
    "Specific practice 4..."
  ],

  "next_medical_steps": [
    "Specific action step 1",
    "Specific action step 2...",
    "Specific action step 3..."
  ],

  "caregiver_guidance": "2-3 sentences of practical guidance for 
    family members and caregivers. Write null if stage is NoImpairment.",

  "progression_risk": "2-3 sentences describing typical progression 
    timeline without intervention, and key factors that can slow it down. 
    Write null if stage is NoImpairment.",

  "questions_for_doctor": [
    "Specific question 1 the patient should ask their neurologist?",
    "Specific question 2?",
    "Specific question 3?",
    "Specific question 4?",
    "Specific question 5?"
  ]
}

STRICT RULES:
- Write directly TO the patient using 'you' and 'your'
- Use grade 8 reading level or simpler — no Latin terms, no jargon
- NEVER suggest specific drug names or dosages
- ALWAYS end next_medical_steps with recommending a qualified neurologist
- Keep each list item to 1-2 sentences maximum
- Output ONLY the JSON — nothing else
- Every single field MUST be specific to the exact stage and confidence 
  level provided — never give generic advice that could apply to any stage
"""


# ── Per-stage prompt context injected at call time ────────────
STAGE_CONTEXT = {
    "NoImpairment": {
        "tone": "reassuring and celebratory but not dismissive",
        "focus": "this is genuinely good news — the scan shows no signs of Alzheimer's-related changes. Focus on maintaining brain health, prevention strategies, and the relief this result should bring. Do NOT give generic memory tips that imply concern — this person's brain looks healthy.",
        "possible_reasons": "reasons the brain appears healthy: genetics, lifestyle, diet, exercise history, cognitive engagement, sleep habits",
        "symptoms_to_watch": "very subtle early warning signs to be aware of ONLY as a precaution — things like unusual word-finding difficulty, getting lost in familiar places, or personality changes. Frame these as 'keep in mind, not as current concerns'",
        "diet": "brain-protective diet for someone with a healthy scan — Mediterranean diet, omega-3s, antioxidants to MAINTAIN this good result",
        "lifestyle": "activities proven to keep a healthy brain healthy long-term — physical exercise, social connection, sleep quality, mental stimulation",
        "next_steps": "routine follow-up, no urgency. Perhaps a baseline cognitive assessment for future comparison. Regular GP check-ins.",
        "caregiver": "null — no caregiver guidance needed for a healthy result",
        "progression": "null — no progression risk to discuss",
        "doctor_questions": "questions about maintaining brain health, how often to re-scan, family history implications, preventive supplements"
    },
    "VeryMildImpairment": {
        "tone": "calm, warm, and measured — this is early stage and there is real hope",
        "focus": "very early, subtle changes detected that may or may not progress. This is the best time to act. The changes are small and may not affect daily life yet.",
        "possible_reasons": "early age-related changes, early Alzheimer's pathology, vascular factors, sleep disruption, chronic stress, metabolic factors like diabetes or high blood pressure",
        "symptoms_to_watch": "very mild symptoms: occasionally forgetting recent conversations, misplacing items more than usual, slight difficulty finding words, taking longer to recall names",
        "diet": "MIND diet specifically shown to slow early cognitive decline — leafy greens daily, berries, nuts, olive oil, fish twice a week, limiting red meat and processed foods",
        "lifestyle": "high-impact interventions at this early stage: 150 min/week aerobic exercise (proven to slow progression), quality sleep 7-8 hours, stress management, social engagement, learning new skills",
        "next_steps": "neurologist referral within 3 months, cognitive baseline testing, blood work to rule out reversible causes (thyroid, B12, sleep apnea screening)",
        "caregiver": "gentle awareness guidance for family — what to observe, how to be supportive without being overprotective",
        "progression": "brief, honest but hopeful — early intervention at this stage has the strongest evidence for slowing progression",
        "doctor_questions": "questions about reversible causes, cognitive testing, whether medication is appropriate, genetic testing options, clinical trial eligibility"
    },
    "MildImpairment": {
        "tone": "direct, honest, compassionate — the situation is real but manageable with the right support",
        "focus": "noticeable memory and thinking changes that are affecting some daily activities. This needs active medical management now. Be honest about the reality while emphasising that much can be done.",
        "possible_reasons": "Alzheimer's disease pathology, mixed dementia (Alzheimer's plus vascular), other neurodegenerative conditions, medication side effects, depression mimicking cognitive decline",
        "symptoms_to_watch": "clear symptoms: difficulty managing finances or medications independently, getting confused on familiar routes, repeating questions in the same conversation, struggling with complex tasks like cooking a full meal, mood and personality shifts",
        "diet": "strict brain-protective diet now therapeutic: Mediterranean-DASH Intervention for Neurodegenerative Delay (MIND diet), reducing alcohol completely, adequate protein for brain maintenance, hydration",
        "lifestyle": "structured routine is now essential: supervised exercise, cognitive rehabilitation exercises, music therapy (shown to help at this stage), maintaining social connections, reducing isolation",
        "next_steps": "urgent neurologist referral within 4-6 weeks, formal neuropsychological testing, brain MRI with contrast if not done, medication evaluation, driving assessment, advance care planning discussion",
        "caregiver": "practical caregiver guidance — safety in the home, medication management, how to communicate effectively with someone at this stage, caregiver burnout awareness",
        "progression": "honest progression timeline, factors that can slow decline, importance of early intervention now",
        "doctor_questions": "questions about medication options, what to expect over next 1-2 years, driving safety, legal and financial planning, support services available"
    },
    "ModerateImpairment": {
        "tone": "urgent, compassionate, and supportive — this is serious and the family needs to be involved",
        "focus": "significant cognitive decline affecting most daily activities. Full-time support is likely needed. Be clear and direct while remaining compassionate. The patient needs immediate comprehensive care.",
        "possible_reasons": "advanced Alzheimer's disease, mixed dementia, other late-stage neurodegenerative conditions. At this stage causes are well-established.",
        "symptoms_to_watch": "significant symptoms: confusion about time, place, or people; needing help with bathing, dressing, and eating; wandering behavior; significant personality changes; sleep-wake cycle disruption; difficulty recognising family members",
        "diet": "nutrition support is now critical: ensuring adequate caloric intake (weight loss is common), soft foods if swallowing is affected, regular meal schedule, monitoring hydration as patients may not recognise thirst",
        "lifestyle": "structured, supervised daily routine; familiar music from their past (proven to calm and connect); gentle physical activity with assistance; maintaining dignity in all personal care; minimising environmental changes that cause confusion",
        "next_steps": "immediate neurologist appointment — within 2 weeks, medication review for symptom management, occupational therapy home safety assessment, social worker referral for care planning, legal documents (power of attorney, advance directives) if not in place",
        "caregiver": "detailed, practical caregiver guidance — this is now a full caregiving situation. Safety in the home, how to handle difficult behaviours, respite care options, caregiver support groups, when to consider professional care facilities",
        "progression": "honest but compassionate — this is an advanced stage. Focus on quality of life, comfort, and dignity rather than reversal",
        "doctor_questions": "questions about specialist memory clinic referral, medication for behavioural symptoms, palliative care planning, what to expect in coming months, how to access community support services"
    }
}


# ── Gemini setup — build client ONCE at startup ───────────────
gemini_available = False
gemini_client    = None
if GEMINI_API_KEY and GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY_HERE':
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_client = genai.GenerativeModel(
            model_name         = GEMINI_MODEL,
            system_instruction = SYSTEM_PROMPT,
            generation_config  = genai.GenerationConfig(
                temperature        = 0.4,
                response_mime_type = "application/json",
                max_output_tokens  = 1400,
            )
        )
        gemini_available = True
        print(f"  Gemini API     : Ready ({GEMINI_MODEL})")
    except ImportError:
        print("  Gemini API     : SKIPPED (run: pip install google-generativeai)")
else:
    print("  Gemini API     : SKIPPED (add your API key to GEMINI_API_KEY in app.py)")

print(f"\n  Ready at http://localhost:5000")
print("="*55 + "\n")



def call_gemini(class_name, confidence, probabilities):
    """
    Calls Gemini with a richly contextualised, stage-specific prompt.
    Returns parsed JSON dict or None on failure.
    """
    if not gemini_available or gemini_client is None:
        return None

    response = None
    try:
        # Confidence descriptor
        if confidence >= 0.90:   conf_desc = "very high confidence"
        elif confidence >= 0.75: conf_desc = "high confidence"
        elif confidence >= 0.60: conf_desc = "moderate confidence"
        else:                    conf_desc = "borderline — the model was uncertain between stages"

        # Probability breakdown
        prob_lines = "\n".join([
            f"  {CLASS_NAMES[i]}: {p*100:.1f}%"
            for i, p in enumerate(probabilities)
        ])

        # Second-highest class (what it almost was)
        sorted_probs = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)
        runner_up_idx, runner_up_prob = sorted_probs[1]
        runner_up_name = CLASS_NAMES[runner_up_idx]

        # Compact stage guidance — only the essentials Gemini needs
        ctx = STAGE_CONTEXT.get(class_name, STAGE_CONTEXT["VeryMildImpairment"])

        uncertainty_note = (
            f"This is a clear result."
            if confidence >= 0.85 else
            f"Moderate uncertainty — runner-up was {runner_up_name} at {runner_up_prob*100:.1f}%. Mention this briefly in what_this_means."
        )

        healthy = class_name == "NoImpairment"

        text_prompt = (
            f"Stage: {class_name} | Confidence: {confidence*100:.1f}% ({conf_desc})\n"
            f"Probabilities: {', '.join(f'{CLASS_NAMES[i]} {p*100:.0f}%' for i,p in enumerate(probabilities))}\n"
            f"Tone: {ctx['tone']}\n"
            f"Focus: {ctx['focus']}\n"
            f"Confidence note: {uncertainty_note}\n"
            f"{'caregiver_guidance and progression_risk MUST be null.' if healthy else 'Include caregiver_guidance and progression_risk.'}\n"
            f"{'This is HEALTHY — no disease management advice.' if healthy else 'Be honest about cognitive changes.'}\n"
            f"All content must be SPECIFIC to {class_name} — no generic advice.\n"
            f"Generate JSON now."
        )

        t0       = time.time()
        response = gemini_client.generate_content(text_prompt)
        print(f"     Gemini: {time.time()-t0:.1f}s")

        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        return json.loads(raw)

    except json.JSONDecodeError as e:
        print(f"     Gemini JSON error: {e} | {(response.text[:200] if response else '')}")
        return None
    except Exception as e:
        print(f"     Gemini error: {e}")
        return None


# ── Grad-CAM helpers ──────────────────────────────────────────
def predict_and_gradcam(img_batch, skip_gradcam=False):
    img_tensor = tf.constant(img_batch, dtype=tf.float32)

    if skip_gradcam:
        _, preds_raw = grad_model(img_tensor, training=False)
        return np.array(preds_raw).flatten(), None

    # ── Correct GradCAM: watch the conv layer output, not input ──
    # Watching the input causes near-zero gradients flowing back
    # through many layers — the conv output is the right anchor.
    with tf.GradientTape() as tape:
        conv_out, preds_raw = grad_model(img_tensor, training=False)
        if isinstance(preds_raw, (list, tuple)):
            preds_raw = tf.cast(preds_raw[0], tf.float32)
        else:
            preds_raw = tf.cast(preds_raw, tf.float32)

        tape.watch(conv_out)           # watch conv activations, not raw input
        cidx        = int(tf.argmax(preds_raw[0]))
        class_score = preds_raw[:, cidx]

    grads = tape.gradient(class_score, conv_out)  # shape: (1, H, W, C)

    # ── Robust pooling: use ReLU on grads before averaging ───────
    # Plain mean collapses when negative grads cancel positives.
    # Only keep channels where the gradient is positively activated.
    grads_relu   = tf.nn.relu(grads)                        # (1,H,W,C)
    pooled_grads = tf.reduce_mean(grads_relu, axis=(0,1,2)) # (C,)

    # Weighted sum of conv feature maps
    heatmap = tf.reduce_sum(conv_out[0] * pooled_grads, axis=-1)  # (H,W)
    heatmap = tf.nn.relu(heatmap).numpy()                          # keep positives

    # ── Percentile normalisation — stretches contrast properly ───
    # Dividing by max() fails when activations cluster in a tiny
    # range (whole map stays near 0 → all blue in JET).
    # Stretching between the 5th and 95th percentile forces the
    # full colour range to be used wherever there IS activation.
    p5  = np.percentile(heatmap, 5)
    p95 = np.percentile(heatmap, 95)
    if p95 - p5 > 1e-6:
        heatmap = np.clip((heatmap - p5) / (p95 - p5), 0, 1)
    else:
        # Fallback: flat map — use simple max norm
        heatmap = heatmap / (np.max(heatmap) + 1e-8)

    return np.array(preds_raw).flatten(), heatmap


def overlay_heatmap(rgb, heatmap, alpha=0.50):
    H, W = rgb.shape[:2]
    hmap = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_CUBIC)

    # Slight gaussian smoothing so the overlay isn't blocky
    hmap = cv2.GaussianBlur(hmap, (11, 11), 0)

    # Re-normalise after blur
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)

    hcol = cv2.applyColorMap(np.uint8(255 * hmap), cv2.COLORMAP_JET)
    hrgb = cv2.cvtColor(hcol, cv2.COLOR_BGR2RGB)

    # Blend: stronger alpha so activated regions are clearly visible
    return cv2.addWeighted(rgb, 1 - alpha, hrgb, alpha, 0)

def to_b64(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format='JPEG', quality=88, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


# ── /predict ─────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template("alzheimer_detection_ui.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file. POST multipart/form-data with key "file"'}), 400
    try:
        t_start = time.time()

        # ── Decode image ──────────────────────────────────────
        raw   = request.files['file'].read()
        pil   = Image.open(io.BytesIO(raw)).convert('RGB').resize(IMG_SIZE, Image.BILINEAR)
        arr   = np.array(pil, dtype=np.float32) / 255.0
        batch = np.expand_dims(arr, axis=0)
        t_img = time.time() - t_start
        print(f"     Image decode: {t_img*1000:.0f}ms")

        # ── Single TF forward pass (always includes GradCAM) ─
        # Running GradCAM every time is only ~30ms more than
        # a plain predict, but saves an entire second pass for
        # impairment cases — net win for the common case.
        t_tf = time.time()
        preds, heatmap = predict_and_gradcam(batch, skip_gradcam=False)
        print(f"     TF inference: {(time.time()-t_tf)*1000:.0f}ms")

        cidx           = int(np.argmax(preds))
        conf           = float(preds[cidx])
        has_impairment = (cidx != NO_IMPAIRMENT_IDX)
        print(f"  → [{cidx}] {CLASS_NAMES[cidx]}  ({conf*100:.1f}%)")

        # ── Submit Gemini immediately (parallel with GradCAM render) ──
        skip_ai = request.form.get('skip_ai', 'false') == 'true'
        gemini_future = _executor.submit(
            call_gemini, CLASS_NAMES[cidx], conf, preds.tolist()
        ) if not skip_ai else _executor.submit(lambda: None)
        # ── GradCAM overlay (only for impairment) ─────────────
        gradcam_b64 = None
        if has_impairment and heatmap is not None:
            orig        = np.array(pil, dtype=np.uint8)
            gradcam_b64 = to_b64(overlay_heatmap(orig, heatmap))
            print(f"     Grad-CAM rendered")

        # ── Collect Gemini (likely already done) ──────────────
        ai_report = gemini_future.result(timeout=14)

        print(f"     Total: {time.time()-t_start:.2f}s")

        return jsonify({
            'class_idx'     : cidx,
            'class_name'    : CLASS_NAMES[cidx],
            'confidence'    : conf,
            'probabilities' : preds.tolist(),
            'gradcam_image' : gradcam_b64,
            'has_impairment': has_impairment,
            'ai_report'     : ai_report,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500, 500


@app.route('/health')
def health():
    return jsonify({
        'status'        : 'ok',
        'model'         : MODEL_PATH,
        'gradcam_layer' : last_conv,
        'classes'       : CLASS_NAMES,
        'gemini_enabled': gemini_available
    })

@app.route('/debug')
def debug():
    rows = ''
    for i, name in enumerate(CLASS_NAMES):
        note = ('<span style="color:#00ff9d">&#x2190; NoImpairment</span>'
                if i == NO_IMPAIRMENT_IDX
                else '<span style="color:#ffd166">Impairment class</span>')
        rows += f'<tr><td>{i}</td><td>{name}</td><td>{note}</td></tr>\n'
    html = (
        '<html><head><style>body{font-family:monospace;background:#050a12;color:#e2edf8;padding:40px;}'
        'h2{color:#00c8ff;}table{border-collapse:collapse;width:620px;}'
        'td,th{padding:10px 16px;border:1px solid rgba(0,200,255,0.2);text-align:left;}'
        'th{color:#00c8ff;background:rgba(0,200,255,0.05);}.note{color:#637a8f;font-size:13px;line-height:1.8;}'
        'code{background:#111d2e;padding:2px 8px;border-radius:4px;color:#00c8ff;}'
        '</style></head><body>'
        f'<h2>NeuroScan AI &#8212; Debug</h2>'
        f'<p class="note">Gemini AI: <b style="color:{"#00ff9d" if gemini_available else "#ff3b5c"}">'
        f'{"Enabled" if gemini_available else "Disabled — add API key to app.py"}</b></p>'
        '<br><table><tr><th>Index</th><th>Class Name</th><th>Notes</th></tr>'
        + rows +
        '</table><br><p><a href="/" style="color:#00c8ff">&#8592; Back to App</a></p>'
        '</body></html>'
    )
    return html


# ── /report — generate downloadable PDF ──────────────────────
@app.route('/report', methods=['POST', 'OPTIONS'])
def generate_report():
    if request.method == 'OPTIONS':
        r = jsonify({'ok': True})
        r.headers["Access-Control-Allow-Origin"]  = "*"
        r.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        r.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return r, 200

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, Image as RLImage, KeepTogether
        )
        from reportlab.platypus import PageBreak
        import datetime

        data       = request.get_json()
        class_name = data.get('class_name', 'Unknown')
        confidence = data.get('confidence', 0)
        probs      = data.get('probabilities', [0, 0, 0, 0])
        ai_report  = data.get('ai_report', {})
        gradcam_b64= data.get('gradcam_image')
        scan_date  = data.get('scan_date', datetime.datetime.now().strftime('%B %d, %Y'))

        # ── Colour palette — matches light UI theme ─────────
        NAVY       = colors.HexColor('#0f2952')   # --navy
        BLUE       = colors.HexColor('#1e5ba8')   # --blue
        CYAN       = colors.HexColor('#0ea5e9')   # --cyan
        GREEN      = colors.HexColor('#10b981')   # --green
        YELLOW     = colors.HexColor('#eab308')   # --yellow
        ORANGE     = colors.HexColor('#f97316')   # --orange
        RED        = colors.HexColor('#ef4444')   # --red
        LIGHT_GREY = colors.HexColor('#f7f9fc')   # --surface2
        MID_GREY   = colors.HexColor('#64748b')   # --text-dim
        MUTED      = colors.HexColor('#94a3b8')   # --text-muted
        DARK_TEXT  = colors.HexColor('#0f2952')   # --text (same as navy)
        WHITE      = colors.white
        BORDER     = colors.HexColor('#dde8f0')   # --border solid

        sev_map = {
            'NoImpairment'      : ('NORMAL',   GREEN,  '#ecfdf5', '#065f46'),
            'VeryMildImpairment': ('VERY MILD', YELLOW, '#fefce8', '#854d0e'),
            'MildImpairment'    : ('MILD',      ORANGE, '#fff7ed', '#9a3412'),
            'ModerateImpairment': ('MODERATE',  RED,    '#fef2f2', '#991b1b'),
        }
        sev_tuple   = sev_map.get(class_name, ('UNKNOWN', MID_GREY, '#f7f9fc', '#475569'))
        badge_text  = sev_tuple[0]
        sev_color   = sev_tuple[1]
        sev_bg      = sev_tuple[2]
        sev_text_c  = colors.HexColor(sev_tuple[3])  # dark readable text on light badge bg

        # ── Build PDF in memory ───────────────────────────────
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize    = A4,
            leftMargin  = 18*mm,
            rightMargin = 18*mm,
            topMargin   = 14*mm,
            bottomMargin= 16*mm,
            title       = 'NeuroScan AI — MRI Analysis Report',
            author      = 'NeuroScan AI',
        )

        W = A4[0] - 36*mm   # usable width

        # ── Paragraph styles ──────────────────────────────────
        def ps(name, font='Helvetica', size=10, color=DARK_TEXT,
               leading=None, align=TA_LEFT, spaceBefore=0, spaceAfter=4,
               bold=False):
            return ParagraphStyle(
                name,
                fontName    = f'Helvetica-Bold' if bold else font,
                fontSize    = size,
                textColor   = color,
                leading     = leading or size * 1.45,
                alignment   = align,
                spaceBefore = spaceBefore,
                spaceAfter  = spaceAfter,
            )

        S_TITLE       = ps('title',    size=22, color=NAVY,      bold=True,  leading=26, align=TA_CENTER)
        S_SUBTITLE    = ps('subtitle', size=10, color=MID_GREY,  align=TA_CENTER, spaceAfter=2)
        S_SEC_HEAD    = ps('sechead',  size=10, color=NAVY,      bold=True,  spaceBefore=14, spaceAfter=5)
        S_BODY        = ps('body',     size=9.5,color=DARK_TEXT, leading=15, spaceAfter=4, align=TA_JUSTIFY)
        S_BULLET      = ps('bullet',   size=9.5,color=DARK_TEXT, leading=15, spaceAfter=3)
        S_LABEL       = ps('label',    size=8,  color=MID_GREY,  bold=False, spaceAfter=1)
        S_VALUE       = ps('value',    size=11, color=DARK_TEXT, bold=True,  spaceAfter=0)
        S_BADGE       = ps('badge',    size=13, color=WHITE,     bold=True,  align=TA_CENTER)
        S_DISCLAIMER  = ps('disc',     size=7.5,color=MID_GREY,  leading=11, align=TA_JUSTIFY)
        S_CAPTION     = ps('caption',  size=8,  color=MID_GREY,  align=TA_CENTER, spaceBefore=2)
        S_SMALL_HEAD  = ps('smhead',   size=8.5,color=sev_color, bold=True,  spaceAfter=3)

        story = []

        # ════════════════════════════════════════════════════
        # PAGE 1 — HEADER
        # ════════════════════════════════════════════════════

        # Top header bar — navy bg, date RIGHT-aligned
        header_data = [[
            Paragraph('<font color="white"><b>NEUROSCAN AI</b></font>',
                      ps('h1', size=16, color=WHITE, bold=True)),
            Paragraph(f'<font color="#7dd3fc">MRI Neurological Analysis Report</font>',
                      ps('h2', size=10, color=CYAN, align=TA_CENTER)),
            Paragraph(f'<font color="#94a3b8">{scan_date}</font>',
                      ps('h3', size=9, color=MUTED, align=TA_RIGHT)),
        ]]
        header_table = Table(header_data, colWidths=[W*0.32, W*0.40, W*0.28])
        header_table.setStyle(TableStyle([
            ('BACKGROUND',   (0,0), (-1,-1), NAVY),
            ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING',  (0,0), (-1,-1), 14),
            ('RIGHTPADDING', (0,0), (-1,-1), 14),
            ('TOPPADDING',   (0,0), (-1,-1), 14),
            ('BOTTOMPADDING',(0,0), (-1,-1), 14),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 6*mm))

        # ── Diagnosis summary card ────────────────────────────
        # Badge cell
        # Badge uses dark readable text on light sev background
        S_BADGE_NEW = ps('badge2', size=12, color=sev_text_c, bold=True, align=TA_CENTER)
        badge_cell = Table(
            [[Paragraph(badge_text, S_BADGE_NEW)]],
            colWidths=[34*mm]
        )
        badge_cell.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), colors.HexColor(sev_bg)),
            ('TOPPADDING',    (0,0),(-1,-1), 10),
            ('BOTTOMPADDING', (0,0),(-1,-1), 10),
            ('LEFTPADDING',   (0,0),(-1,-1), 6),
            ('RIGHTPADDING',  (0,0),(-1,-1), 6),
            ('LINEAFTER',     (0,0),(0,-1),  1, sev_color),
        ]))

        # Add space before "Impairment" in class name display
        def fmt_class(cn):
            import re
            return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', cn)

        conf_pct = f'{confidence*100:.1f}%'
        S_LABEL2 = ps('lbl2', size=7.5, color=MUTED,     bold=False, spaceAfter=3, align=TA_LEFT)
        S_VAL_SM = ps('vsm',  size=10,  color=DARK_TEXT,  bold=True,  spaceAfter=0)
        S_VAL_CN = ps('vcn',  size=10,  color=DARK_TEXT,  bold=True,  spaceAfter=0)
        S_VAL_CF = ps('vcf',  size=18,  color=sev_text_c, bold=True,  spaceAfter=0)
        summary_inner = Table([
            [Paragraph('DIAGNOSIS',  S_LABEL2),
             Paragraph('CONFIDENCE', S_LABEL2),
             Paragraph('SCAN DATE',  S_LABEL2)],
            [Paragraph(fmt_class(class_name), S_VAL_CN),
             Paragraph(conf_pct,              S_VAL_CF),
             Paragraph(scan_date,             S_VAL_SM)],
        ], colWidths=[W*0.36, W*0.26, W*0.38 - 34*mm])
        summary_inner.setStyle(TableStyle([
            ('VALIGN',        (0,0),(-1,-1), 'TOP'),
            ('LEFTPADDING',   (0,0),(-1,-1), 0),
            ('RIGHTPADDING',  (0,0),(-1,-1), 4),
            ('TOPPADDING',    (0,0),(-1,-1), 0),
            ('BOTTOMPADDING', (0,0),(-1,-1), 0),
            ('LINEBELOW',     (0,0),(-1,0),  0.5, BORDER),
            ('TOPPADDING',    (0,1),(-1,1),  5),
        ]))

        summary_card = Table(
            [[badge_cell, summary_inner]],
            colWidths=[34*mm, W - 34*mm]
        )
        summary_card.setStyle(TableStyle([
            ('BACKGROUND',    (0,0),(-1,-1), WHITE),
            ('BOX',           (0,0),(-1,-1), 1,   BORDER),
            ('BACKGROUND',    (0,0),(0,-1),  colors.HexColor(sev_bg)),
            ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
            ('LEFTPADDING',   (1,0),(1,-1),  14),
            ('RIGHTPADDING',  (1,0),(1,-1),  14),
            ('TOPPADDING',    (0,0),(-1,-1), 12),
            ('BOTTOMPADDING', (0,0),(-1,-1), 12),
        ]))
        story.append(summary_card)
        story.append(Spacer(1, 5*mm))

        # ── Probability breakdown table ───────────────────────
        story.append(Paragraph('CLASS PROBABILITY BREAKDOWN', S_SEC_HEAD))
        story.append(HRFlowable(width=W, thickness=0.5, color=BORDER, spaceAfter=4))

        bar_rows = []
        prob_colors = [ORANGE, RED, GREEN, YELLOW]
        for i, (name, p) in enumerate(zip(CLASS_NAMES, probs)):
            pct   = p * 100
            label = name.replace('Impairment', ' Impairment')
            # Bar as filled table cell
            bar_w  = max(2, (W * 0.45) * p)
            bar_tbl = Table([['']], colWidths=[bar_w], rowHeights=[8])
            bar_tbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0),(-1,-1), prob_colors[i]),
            ]))
            is_top = (name == class_name)
            row_bg = colors.HexColor(sev_bg) if is_top else WHITE
            bar_rows.append([
                Paragraph(f'<b>{label}</b>' if is_top else label,
                          ps('pl', size=9, color=sev_color if is_top else DARK_TEXT, bold=is_top)),
                bar_tbl,
                Paragraph(f'<b>{pct:.1f}%</b>' if is_top else f'{pct:.1f}%',
                          ps('pv', size=9, color=sev_color if is_top else DARK_TEXT,
                             bold=is_top, align=TA_LEFT)),
            ])

        prob_table = Table(bar_rows, colWidths=[W*0.32, W*0.52, W*0.16])
        prob_table.setStyle(TableStyle([
            ('VALIGN',       (0,0),(-1,-1), 'MIDDLE'),
            ('LEFTPADDING',  (0,0),(-1,-1), 8),
            ('RIGHTPADDING', (0,0),(-1,-1), 8),
            ('TOPPADDING',   (0,0),(-1,-1), 6),
            ('BOTTOMPADDING',(0,0),(-1,-1), 6),
            ('ROWBACKGROUNDS',(0,0),(-1,-1),[WHITE, LIGHT_GREY]),
            ('BOX',          (0,0),(-1,-1), 0.5, BORDER),
            ('INNERGRID',    (0,0),(-1,-1), 0.3, BORDER),
        ]))
        story.append(prob_table)
        story.append(Spacer(1, 5*mm))

        # ── Grad-CAM image ────────────────────────────────────
        if gradcam_b64:
            story.append(Paragraph('GRAD-CAM BRAIN ACTIVATION HEATMAP', S_SEC_HEAD))
            story.append(HRFlowable(width=W, thickness=0.5, color=BORDER, spaceAfter=4))

            try:
                raw_b64  = gradcam_b64.split(',', 1)[1] if ',' in gradcam_b64 else gradcam_b64
                img_data = base64.b64decode(raw_b64)
                img_buf  = io.BytesIO(img_data)

                img_w    = 72*mm
                img_h    = 72*mm
                rl_img   = RLImage(img_buf, width=img_w, height=img_h)

                legend_items = [
                    [colors.HexColor('#0000FF'), 'Low activation'],
                    [colors.HexColor('#00FFFF'), 'Below average'],
                    [colors.HexColor('#00FF00'), 'Average'],
                    [colors.HexColor('#FFFF00'), 'Above average'],
                    [colors.HexColor('#FF0000'), 'High activation'],
                ]
                legend_rows = []
                for lc, lt in legend_items:
                    swatch = Table([['']], colWidths=[12], rowHeights=[8])
                    swatch.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),lc)]))
                    legend_rows.append([swatch, Paragraph(lt, ps('leg', size=8, color=DARK_TEXT))])

                legend_tbl = Table(legend_rows, colWidths=[16, 60*mm])
                legend_tbl.setStyle(TableStyle([
                    ('VALIGN',      (0,0),(-1,-1),'MIDDLE'),
                    ('LEFTPADDING', (0,0),(-1,-1), 2),
                    ('TOPPADDING',  (0,0),(-1,-1), 3),
                    ('BOTTOMPADDING',(0,0),(-1,-1),3),
                ]))

                explanation = (
                    'Regions highlighted in <b>red/yellow</b> indicate areas of the brain '
                    'that most strongly influenced the AI classification decision. '
                    '<b>Blue</b> regions had minimal influence. '
                    'This visualisation does not replace clinical imaging interpretation.'
                )

                right_col_w = W - img_w - 24  # account for outer box padding
                heatmap_layout = Table(
                    [[rl_img,
                      [Paragraph('Activation Map Legend',
                                 ps('lh', size=9, color=NAVY, bold=True, spaceAfter=8)),
                       legend_tbl,
                       Spacer(1, 10),
                       Paragraph(explanation,
                                 ps('exp', size=8, color=MID_GREY, leading=12,
                                    align=TA_LEFT, spaceAfter=0))]]],
                    colWidths=[img_w, right_col_w]
                )
                heatmap_layout.setStyle(TableStyle([
                    ('VALIGN',        (0,0),(-1,-1), 'TOP'),
                    ('LEFTPADDING',   (0,0),(-1,-1), 0),
                    ('RIGHTPADDING',  (0,0),(-1,-1), 0),
                    ('TOPPADDING',    (0,0),(-1,-1), 0),
                    ('BOTTOMPADDING', (0,0),(-1,-1), 0),
                    ('LEFTPADDING',   (1,0),(1,-1),  14),
                    ('RIGHTPADDING',  (1,0),(1,-1),  4),
                ]))

                heatmap_box = Table([[heatmap_layout]], colWidths=[W])
                heatmap_box.setStyle(TableStyle([
                    ('BOX',           (0,0),(-1,-1), 1,   BORDER),
                    ('BACKGROUND',    (0,0),(-1,-1), LIGHT_GREY),
                    ('LEFTPADDING',   (0,0),(-1,-1), 12),
                    ('RIGHTPADDING',  (0,0),(-1,-1), 12),
                    ('TOPPADDING',    (0,0),(-1,-1), 12),
                    ('BOTTOMPADDING', (0,0),(-1,-1), 12),
                ]))
                story.append(heatmap_box)
                story.append(Spacer(1, 5*mm))
            except Exception as e:
                print(f"     PDF heatmap error: {e}")

        # ════════════════════════════════════════════════════
        # PAGE 2 — AI REPORT
        # ════════════════════════════════════════════════════
        if ai_report:
            story.append(PageBreak())

            # Page 2 header
            p2_header = Table([[
                Paragraph('<font color="white"><b>NEUROSCAN AI</b></font>',
                          ps('p2h', size=14, color=WHITE, bold=True)),
                Paragraph('<font color="#7dd3fc">AI Clinical Assistant Report</font>',
                          ps('p2s', size=10, color=CYAN, align=TA_CENTER)),
                Paragraph(f'<font color="#94a3b8">{fmt_class(class_name)}</font>',
                          ps('p2c', size=9, color=MUTED, align=TA_RIGHT)),
            ]], colWidths=[W*0.32, W*0.40, W*0.28])
            p2_header.setStyle(TableStyle([
                ('BACKGROUND',   (0,0),(-1,-1), NAVY),
                ('VALIGN',       (0,0),(-1,-1), 'MIDDLE'),
                ('LEFTPADDING',  (0,0),(-1,-1), 14),
                ('RIGHTPADDING', (0,0),(-1,-1), 14),
                ('TOPPADDING',   (0,0),(-1,-1), 12),
                ('BOTTOMPADDING',(0,0),(-1,-1), 12),
            ]))
            story.append(p2_header)
            story.append(Spacer(1, 6*mm))

            def section(icon, title, content, list_style=None):
                """Helper to render a report section as a styled card."""
                S_SEC_TITLE = ps('sh', size=9, color=NAVY, bold=True, spaceAfter=0)
                S_SEC_BODY  = ps('sb', size=9.5, color=MID_GREY, leading=14, spaceAfter=3, align=TA_LEFT)
                S_SEC_BULL  = ps('sbull', size=9.5, color=MID_GREY, leading=14, spaceAfter=2)

                header_para = Paragraph(f'{icon}  {title.upper()}', S_SEC_TITLE)
                header_row = Table([[header_para]], colWidths=[W])
                header_row.setStyle(TableStyle([
                    ('BACKGROUND',    (0,0),(-1,-1), LIGHT_GREY),
                    ('LEFTPADDING',   (0,0),(-1,-1), 12),
                    ('RIGHTPADDING',  (0,0),(-1,-1), 12),
                    ('TOPPADDING',    (0,0),(-1,-1), 8),
                    ('BOTTOMPADDING', (0,0),(-1,-1), 8),
                    ('LINEBELOW',     (0,0),(-1,-1), 0.5, BORDER),
                ]))

                body_elems = []
                if isinstance(content, str):
                    body_elems.append(Paragraph(content, S_SEC_BODY))
                elif isinstance(content, list):
                    for item in content:
                        body_elems.append(
                            Paragraph(f'<bullet>\u25b8</bullet> {item}', S_SEC_BULL)
                        )

                body_wrap = Table([[e] for e in body_elems], colWidths=[W - 24])
                body_wrap.setStyle(TableStyle([
                    ('LEFTPADDING',   (0,0),(-1,-1), 12),
                    ('RIGHTPADDING',  (0,0),(-1,-1), 12),
                    ('TOPPADDING',    (0,0),(-1,-1), 4),
                    ('BOTTOMPADDING', (0,0),(-1,-1), 3),
                ]))

                card = Table([[header_row], [body_wrap]], colWidths=[W])
                card.setStyle(TableStyle([
                    ('BOX',           (0,0),(-1,-1), 0.5, BORDER),
                    ('LINEBEFORE',    (0,0),(-1,-1), 3,   CYAN),
                    ('LEFTPADDING',   (0,0),(-1,-1), 0),
                    ('RIGHTPADDING',  (0,0),(-1,-1), 0),
                    ('TOPPADDING',    (0,0),(-1,-1), 0),
                    ('BOTTOMPADDING', (0,0),(-1,-1), 0),
                ]))
                story.append(KeepTogether([card, Spacer(1, 3.5*mm)]))

            sections_def = [
                ('🔍', 'What This Result Means',        ai_report.get('what_this_means')),
                ('🧠', 'Why This May Have Happened',    ai_report.get('possible_reasons')),
                ('⚠', 'Symptoms to Watch For',         ai_report.get('symptoms_to_watch')),
                ('🥗', 'Diet Recommendations',          ai_report.get('diet_recommendations')),
                ('🏃', 'Lifestyle Practices',           ai_report.get('lifestyle_practices')),
                ('👨‍⚕️','Next Medical Steps',            ai_report.get('next_medical_steps')),
                ('🏠', 'Caregiver Guidance',            ai_report.get('caregiver_guidance')),
                ('🔄', 'Progression Risk & Timeline',   ai_report.get('progression_risk')),
                ('❓', 'Questions to Ask Your Doctor',  ai_report.get('questions_for_doctor')),
            ]

            for icon, title, content in sections_def:
                if content:
                    section(icon, title, content)

        # ── Disclaimer footer ─────────────────────────────────
        story.append(Spacer(1, 4*mm))
        story.append(HRFlowable(width=W, thickness=0.5, color=BORDER))
        story.append(Spacer(1, 2*mm))
        disclaimer_text = (
            '<b>IMPORTANT MEDICAL DISCLAIMER:</b> This report is generated by an AI '
            'screening tool for research and educational purposes only. It is NOT a '
            'certified medical diagnosis and must not be used as a substitute for '
            'professional medical advice. All findings must be reviewed and confirmed '
            'by a qualified neurologist or medical professional before any clinical '
            'decisions are made. If you have concerns about your health, please consult '
            'a licensed healthcare provider immediately.'
        )
        story.append(Paragraph(disclaimer_text, S_DISCLAIMER))
        story.append(Spacer(1, 1*mm))
        story.append(Paragraph(
            f'Generated by NeuroScan AI  •  {scan_date}  •  For Research Use Only',
            ps('foot', size=7, color=MID_GREY, align=TA_CENTER)
        ))

        # ── Build and return PDF ──────────────────────────────
        doc.build(story)
        buf.seek(0)
        pdf_bytes = buf.read()
        pdf_b64   = base64.b64encode(pdf_bytes).decode()
        print(f"     PDF generated ({len(pdf_bytes)//1024}KB)")

        return jsonify({'pdf': f'data:application/pdf;base64,{pdf_b64}'})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)