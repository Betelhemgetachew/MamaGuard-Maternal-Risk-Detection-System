"""
SMS Service — Africa's Talking Gateway
Falls back to a local mock when AT credentials are not configured.
"""

import os
from database import log_sms

# Load from environment (set in .env or system env)
AT_USERNAME = os.getenv("AT_USERNAME", "sandbox")
AT_API_KEY  = os.getenv("AT_API_KEY",  "")          # Leave blank → mock mode


# ── Message Templates (safe, supportive — NOT clinical) ───────────────────────

TEMPLATES = {
    "appointment_reminder": (
        "Hi {name}! 👋 Just a friendly reminder that your antenatal visit is "
        "scheduled for {date}. We look forward to seeing you. Remember to "
        "bring your health card. Stay strong, mama! 💪"
    ),
    "post_visit_support": (
        "Hi {name}! Great seeing you today 🌟 Remember to drink plenty of "
        "water and take your iron supplement before bed. Your next visit is "
        "on {next_date}. Take care!"
    ),
    "wellness_tip": (
        "Hi {name}! Daily tip: Eat a colourful plate — leafy greens & beans "
        "are great for you and baby. Keep going, you're doing amazing! 🥦❤️"
    ),
    "missed_appointment": (
        "Hi {name}, we missed you at your appointment today! No worries — "
        "please call us to reschedule. Your health matters to us. "
        "Call: {clinic_phone}"
    ),
    "nurse_alert": (
        "⚠️ NURSE ALERT: Patient {name} (ID: {patient_id}) has missed "
        "{missed_count} consecutive appointments. Please initiate "
        "human follow-up immediately."
    ),
}

CLINIC_PHONE = os.getenv("CLINIC_PHONE", "+254700000000")


def _send_via_at(phone: str, message: str) -> tuple[bool, str]:
    """Send via Africa's Talking SDK."""
    try:
        import africastalking
        africastalking.initialize(AT_USERNAME, AT_API_KEY)
        sms = africastalking.SMS
        response = sms.send(message, [phone])
        status = response["SMSMessageData"]["Recipients"][0]["status"]
        return status == "Success", status
    except Exception as e:
        return False, str(e)


def _send_mock(phone: str, message: str) -> tuple[bool, str]:
    """Local mock — prints to stdout, always succeeds."""
    print(f"\n📱 [MOCK SMS] → {phone}")
    print(f"   {message}")
    return True, "mock_success"


def send_sms(patient_id: str, phone: str, message_type: str,
             message_body: str) -> bool:
    """Route SMS through AT or mock, then log result."""
    if AT_API_KEY and AT_USERNAME != "sandbox":
        ok, status = _send_via_at(phone, message_body)
    else:
        ok, status = _send_mock(phone, message_body)

    log_sms(patient_id, phone, message_type, message_body,
            status="sent" if ok else "failed")
    return ok


# ── High-level send functions ─────────────────────────────────────────────────

def send_post_visit_sms(patient_id: str, name: str, phone: str,
                         next_visit_date: str) -> bool:
    body = TEMPLATES["post_visit_support"].format(
        name=name.split()[0], next_date=next_visit_date
    )
    return send_sms(patient_id, phone, "post_visit_support", body)


def send_appointment_reminder(patient_id: str, name: str, phone: str,
                               appt_date: str) -> bool:
    body = TEMPLATES["appointment_reminder"].format(
        name=name.split()[0], date=appt_date
    )
    return send_sms(patient_id, phone, "appointment_reminder", body)


def send_missed_appointment_sms(patient_id: str, name: str,
                                  phone: str) -> bool:
    body = TEMPLATES["missed_appointment"].format(
        name=name.split()[0], clinic_phone=CLINIC_PHONE
    )
    return send_sms(patient_id, phone, "missed_appointment", body)


def send_nurse_alert(nurse_phone: str, patient_name: str,
                      patient_id: str, missed_count: int) -> bool:
    body = TEMPLATES["nurse_alert"].format(
        name=patient_name, patient_id=patient_id, missed_count=missed_count
    )
    return send_sms("SYSTEM", nurse_phone, "nurse_alert", body)


def send_wellness_tip(patient_id: str, name: str, phone: str) -> bool:
    body = TEMPLATES["wellness_tip"].format(name=name.split()[0])
    return send_sms(patient_id, phone, "wellness_tip", body)
