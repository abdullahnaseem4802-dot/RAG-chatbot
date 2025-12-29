"""Mock RAG for local testing - No database needed"""

TREATMENTS = {
    'mosquito': {
        'urdu': """مچھر کا علاج - Eastern Services

علاج: مکمل مچھر کنٹرول (لاروی سائیڈ اور ایڈلٹی سائیڈ)
قیمت: روپے 3,000 - 5,000 فی علاج
کوریج: اندرونی اور بیرونی علاج
گارنٹی: 30 دن کی وارنٹی
فالو اپ: وارنٹی کے دوران مفت معائنہ

رابطہ: +92 336 1101234""",
        'english': """Mosquito Treatment - Eastern Services

Treatment: Comprehensive mosquito control (larvicide + adulticide)
Price: Rs. 3,000 - 5,000 per treatment
Coverage: Indoor and outdoor treatment
Guarantee: 30-day warranty
Follow-up: Free inspection within warranty period

Contact: +92 336 1101234""",
        'roman_urdu': """Machhar ka Ilaaj - Eastern Services

Ilaaj: Mukammal machhar control (larvicide aur adulticide)
Qeemat: Rs. 3,000 - 5,000 per treatment
Coverage: Androni aur baironi ilaaj
Guarantee: 30 din ki warranty
Follow-up: Warranty ke dauran muft muayana

Rabta: +92 336 1101234"""
    },
    'termite': {
        'urdu': """دیمک کا علاج - Eastern Services

علاج: مکمل دیمک پروفنگ کیمیکل بیریئر کے ساتھ
قیمت: روپے 15,000 - 25,000 (رقبے پر منحصر)
کوریج: مکمل پراپرٹی کا علاج
گارنٹی: 5 سال کی وارنٹی
فالو اپ: سالانہ معائنے شامل

رابطہ: +92 336 1101234""",
        'english': """Termite Treatment - Eastern Services

Treatment: Complete termite proofing with chemical barrier
Price: Rs. 15,000 - 25,000 (depends on area)
Coverage: Full property treatment
Guarantee: 5-year warranty
Follow-up: Annual inspections included

Contact: +92 336 1101234""",
        'roman_urdu': """Deemak ka Ilaaj - Eastern Services

Ilaaj: Mukammal deemak proofing chemical barrier ke saath
Qeemat: Rs. 15,000 - 25,000 (raqbe par munhasir)
Coverage: Mukammal property ka ilaaj
Guarantee: 5 saal ki warranty
Follow-up: Saalana muayne shamil

Rabta: +92 336 1101234"""
    },
    'cockroach': {
        'urdu': """کاکروچ کا علاج - Eastern Services

علاج: جیل بیٹنگ اور سپرے ٹریٹمنٹ
قیمت: روپے 2,500 - 4,000 فی علاج
کوریج: کچن، باتھ رومز، اور تمام متاثرہ علاقے
گارنٹی: 60 دن کی وارنٹی
فالو اپ: ضرورت پڑنے پر مفت دوبارہ علاج

رابطہ: +92 336 1101234""",
        'english': """Cockroach Treatment - Eastern Services

Treatment: Gel baiting and spray treatment
Price: Rs. 2,500 - 4,000 per treatment
Coverage: Kitchen, bathrooms, and all affected areas
Guarantee: 60-day warranty
Follow-up: Free re-treatment if needed

Contact: +92 336 1101234""",
        'roman_urdu': """Cockroach ka Ilaaj - Eastern Services

Ilaaj: Gel baiting aur spray treatment
Qeemat: Rs. 2,500 - 4,000 per treatment
Coverage: Kitchen, bathrooms, aur tamam mutasra ilaqe
Guarantee: 60 din ki warranty
Follow-up: Zaroorat parne par muft dobara ilaaj

Rabta: +92 336 1101234"""
    },
    'rodent': {
        'urdu': """چوہے کا علاج - Eastern Services

علاج: ٹریپنگ اور روڈینٹی سائیڈ پلیسمنٹ
قیمت: روپے 4,000 - 7,000 فی علاج
کوریج: مکمل پراپرٹی کوریج
گارنٹی: 45 دن کی وارنٹی
فالو اپ: 1 ماہ کے لیے ہفتہ وار مانیٹرنگ

رابطہ: +92 336 1101234""",
        'english': """Rodent Treatment - Eastern Services

Treatment: Trapping and rodenticide placement
Price: Rs. 4,000 - 7,000 per treatment
Coverage: Complete property coverage
Guarantee: 45-day warranty
Follow-up: Weekly monitoring for 1 month

Contact: +92 336 1101234""",
        'roman_urdu': """Chuhe ka Ilaaj - Eastern Services

Ilaaj: Trapping aur rodenticide placement
Qeemat: Rs. 4,000 - 7,000 per treatment
Coverage: Mukammal property coverage
Guarantee: 45 din ki warranty
Follow-up: 1 maah ke liye haftawar monitoring

Rabta: +92 336 1101234"""
    }
}

def get_mock_treatment(pest_name, language='english'):
    """Get mock treatment recommendation"""
    pest_data = TREATMENTS.get(pest_name.lower(), {})
    return pest_data.get(language, f"Contact us for {pest_name} treatment: +92 336 1101234")
