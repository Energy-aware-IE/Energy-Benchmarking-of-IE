"""
GoLLIE prompt headers for NER in multiple languages.
Each variable is a string containing the code scaffold
to be prepended before `text = ...; result = [` when building prompts.
"""

# ---------------- German ----------------
GERMAN_HEADER = """\
from dataclasses import dataclass

class Template:
    pass

@dataclass
class PER(Template):
    \"\"\"Deutsche Personennamen. Schließt individuelle Menschen ein (Vorname + Nachname, ggf. Titel).
    Schließe Organisationen und Orte aus.\"\"\"
    mention: str

@dataclass
class ORG(Template):
    \"\"\"Organisationen in deutscher Sprache (Behörden, Firmen, Parteien, Vereine, internationale Org.).
    Schließe Personen- und Ortsnamen aus.\"\"\"
    mention: str

@dataclass
class LOC(Template):
    \"\"\"Ortsnamen in deutscher Sprache (Städte, Länder, Regionen, geografische Orte).
    Schließe Personen und Organisationen aus.\"\"\"
    mention: str
"""

# ---------------- English ----------------
ENGLISH_HEADER = """\
from dataclasses import dataclass

class Template:
    pass

@dataclass
class PER(Template):
    \"\"\"English personal names. Includes individual people (first + last name, possibly titles).
    Exclude organizations and places.\"\"\"
    mention: str

@dataclass
class ORG(Template):
    \"\"\"Organizations in English (companies, government bodies, parties, NGOs, international orgs).
    Exclude persons and locations.\"\"\"
    mention: str

@dataclass
class LOC(Template):
    \"\"\"Geographical names in English (cities, countries, regions, landmarks).
    Exclude persons and organizations.\"\"\"
    mention: str
"""

# ---------------- Chinese ----------------
CHINESE_HEADER = """\
from dataclasses import dataclass

class Template:
    pass

@dataclass
class PER(Template):
    \"\"\"中文的人名，包括个人姓名（姓和名），可能包含头衔。
    不包括组织和地点。\"\"\"
    mention: str

@dataclass
class ORG(Template):
    \"\"\"中文的组织名称（公司、政府机构、政党、非政府组织、国际组织等）。
    不包括人名和地名。\"\"\"
    mention: str

@dataclass
class LOC(Template):
    \"\"\"中文的地名（城市、国家、地区、地理位置）。
    不包括人名和组织。\"\"\"
    mention: str
"""

# ---------------- Arabic ----------------
ARABIC_HEADER = """\
from dataclasses import dataclass

class Template:
    pass

@dataclass
class PER(Template):
    \"\"\"الأسماء الشخصية باللغة العربية. تشمل أسماء الأفراد (الاسم الأول + اسم العائلة، وقد تتضمن الألقاب).
    استبعاد أسماء المنظمات والأماكن.\"\"\"
    mention: str

@dataclass
class ORG(Template):
    \"\"\"أسماء المنظمات باللغة العربية (شركات، هيئات حكومية، أحزاب، منظمات غير حكومية، منظمات دولية).
    استبعاد أسماء الأشخاص والأماكن.\"\"\"
    mention: str

@dataclass
class LOC(Template):
    \"\"\"أسماء الأماكن باللغة العربية (مدن، دول، مناطق، مواقع جغرافية).
    استبعاد الأشخاص والمنظمات.\"\"\"
    mention: str
"""

# ---------------- Bulgarian ----------------
BULGARIAN_HEADER = """\
from dataclasses import dataclass

class Template:
    pass

@dataclass
class PER(Template):
    \"\"\"Лични имена на български език. Включва индивидуални хора (собствено и фамилно име, възможни титли).
    Изключва организации и места.\"\"\"
    mention: str

@dataclass
class ORG(Template):
    \"\"\"Организации на български език (компании, държавни институции, партии, НПО, международни организации).
    Изключва личности и географски имена.\"\"\"
    mention: str

@dataclass
class LOC(Template):
    \"\"\"Географски имена на български език (градове, държави, региони, географски обекти).
    Изключва личности и организации.\"\"\"
    mention: str
"""

# ---------------- French ----------------
FRENCH_HEADER = """\
from dataclasses import dataclass

class Template:
    pass

@dataclass
class PER(Template):
    \"\"\"Noms de personnes en français. Comprend les individus (prénom + nom, éventuellement titres).
    Exclure les organisations et les lieux.\"\"\"
    mention: str

@dataclass
class ORG(Template):
    \"\"\"Organisations en français (entreprises, institutions gouvernementales, partis, ONG, organisations internationales).
    Exclure les personnes et les lieux.\"\"\"
    mention: str

@dataclass
class LOC(Template):
    \"\"\"Noms géographiques en français (villes, pays, régions, lieux géographiques).
    Exclure les personnes et les organisations.\"\"\"
    mention: str
"""

# ---------------- Spanish ----------------
SPANISH_HEADER = """\
from dataclasses import dataclass

class Template:
    pass

@dataclass
class PER(Template):
    \"\"\"Nombres de personas en español. Incluye individuos (nombre + apellido, posiblemente títulos).
    Excluir organizaciones y lugares.\"\"\"
    mention: str

@dataclass
class ORG(Template):
    \"\"\"Organizaciones en español (empresas, organismos gubernamentales, partidos, ONG, organizaciones internacionales).
    Excluir personas y lugares.\"\"\"
    mention: str

@dataclass
class LOC(Template):
    \"\"\"Nombres geográficos en español (ciudades, países, regiones, lugares geográficos).
    Excluir personas y organizaciones.\"\"\"
    mention: str
"""

# ---------------- Russian ----------------
RUSSIAN_HEADER = """\
from dataclasses import dataclass

class Template:
    pass

@dataclass
class PER(Template):
    \"\"\"Личные имена на русском языке. Включает людей (имя + фамилия, возможно отчество и титулы).
    Исключать организации и места.\"\"\"
    mention: str

@dataclass
class ORG(Template):
    \"\"\"Организации на русском языке (компании, государственные органы, партии, НКО, международные организации).
    Исключать личности и географические названия.\"\"\"
    mention: str

@dataclass
class LOC(Template):
    \"\"\"Географические названия на русском языке (города, страны, регионы, географические объекты).
    Исключать личности и организации.\"\"\"
    mention: str
"""

# ---------------- Hindi ----------------
HINDI_HEADER = """\
from dataclasses import dataclass

class Template:
    pass

@dataclass
class PER(Template):
    \"\"\"हिंदी में व्यक्तिगत नाम। व्यक्तियों को शामिल करता है (पहला नाम + उपनाम, संभवतः उपाधियाँ)।
    संगठन और स्थान को बाहर रखें।\"\"\"
    mention: str

@dataclass
class ORG(Template):
    \"\"\"हिंदी में संगठन (कंपनियाँ, सरकारी निकाय, दल, गैर-सरकारी संगठन, अंतर्राष्ट्रीय संगठन)।
    व्यक्तियों और स्थानों को बाहर रखें।\"\"\"
    mention: str

@dataclass
class LOC(Template):
    \"\"\"हिंदी में भौगोलिक नाम (शहर, देश, क्षेत्र, भौगोलिक स्थान)।
    व्यक्तियों और संगठनों को बाहर रखें।\"\"\"
    mention: str
"""

# ---------------- Korean ----------------
KOREAN_HEADER = """\
from dataclasses import dataclass

class Template:
    pass

@dataclass
class PER(Template):
    \"\"\"한국어 인명. 개인(이름 + 성, 가능한 경우 직함 포함)을 포함합니다.
    조직과 장소는 제외합니다.\"\"\"
    mention: str

@dataclass
class ORG(Template):
    \"\"\"한국어 조직명 (기업, 정부 기관, 정당, NGO, 국제 조직).
    인명과 지명은 제외합니다.\"\"\"
    mention: str

@dataclass
class LOC(Template):
    \"\"\"한국어 지명 (도시, 국가, 지역, 지리적 위치).
    인명과 조직명은 제외합니다.\"\"\"
    mention: str
"""

# ---------------- Japanese ----------------
JAPANESE_HEADER = """\
from dataclasses import dataclass

class Template:
    pass

@dataclass
class PER(Template):
    \"\"\"日本語の人名。個人（姓名、場合によっては敬称・役職）を含みます。
    組織名と地名は除外してください。\"\"\"
    mention: str

@dataclass
class ORG(Template):
    \"\"\"日本語の組織名（企業、政府機関、政党、NGO、国際機関）。
    人名と地名は除外してください。\"\"\"
    mention: str

@dataclass
class LOC(Template):
    \"\"\"日本語の地名（都市、国家、地域、地理的な場所）。
    人名と組織名は除外してください。\"\"\"
    mention: str
"""
