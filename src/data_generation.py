"""Generate a realistic, *messy* procurement dataset.

Real procurement / accounts-payable exports are unstructured: free-text
line descriptions typed by different people, abbreviations, typos, PO and
invoice codes mixed into the text, marketplace vendors (Amazon Business,
IndiaMART ...) that sell everything, and so on.  Company data like this is
confidential, so this script builds a synthetic dataset that reproduces
those problems.  Every record has a ground-truth spend category taken from
`data/taxonomy.json` so we can train and measure the classifiers.

Run:  python -m src.data_generation --n 5000
"""
import argparse
import json
import random
from datetime import date, timedelta

import pandas as pd

from src import config

# ----------------------------------------------------------------------------
# Catalogue: category -> (items, specialist suppliers, typical departments)
# items are (description, min unit price INR, max unit price INR, unit)
# ----------------------------------------------------------------------------
CATALOGUE = {
    "IT Hardware": {
        "description": "Physical computing and networking equipment: laptops, desktops, monitors, servers, peripherals, printers, network devices.",
        "items": [
            ("Dell Latitude 5440 laptop i5 16GB", 62000, 85000, "nos"),
            ("HP EliteBook 840 G10 notebook", 78000, 110000, "nos"),
            ("Lenovo ThinkPad T14 Gen 4", 70000, 98000, "nos"),
            ("Apple MacBook Pro 14 M3", 160000, 210000, "nos"),
            ("Dell P2422H 24 inch monitor", 13000, 17000, "nos"),
            ("LG 27UL500 27in 4K monitor", 21000, 27000, "nos"),
            ("Logitech MX Keys wireless keyboard", 8500, 11000, "nos"),
            ("Logitech M331 silent wireless mouse", 900, 1400, "nos"),
            ("Dell WD19S USB-C docking station", 14000, 19000, "nos"),
            ("HPE ProLiant DL380 Gen11 rack server", 650000, 950000, "nos"),
            ("Dell PowerEdge R760 server", 700000, 1100000, "nos"),
            ("Cisco Catalyst 9200 48-port switch", 180000, 260000, "nos"),
            ("Ubiquiti UniFi U6 Pro wifi access point", 14000, 19000, "nos"),
            ("Fortinet FortiGate 60F firewall appliance", 65000, 90000, "nos"),
            ("Samsung 990 Pro 1TB NVMe SSD", 9000, 13000, "nos"),
            ("Seagate 4TB external hard drive", 9000, 12000, "nos"),
            ("Synology DS923+ NAS enclosure", 52000, 68000, "nos"),
            ("HP LaserJet Pro M404dn printer", 24000, 31000, "nos"),
            ("Epson EcoTank L3250 printer", 13000, 16000, "nos"),
            ("APC Smart-UPS 1500VA", 38000, 52000, "nos"),
            ("Logitech C920 HD webcam", 6500, 8500, "nos"),
            ("Jabra Evolve2 40 USB headset", 9000, 12500, "nos"),
            ("Apple iPhone 15 corporate handset", 70000, 80000, "nos"),
            ("Samsung Galaxy Tab S9 tablet", 68000, 82000, "nos"),
            ("Crucial 16GB DDR5 RAM module", 4200, 6000, "nos"),
            ("Lenovo ThinkCentre M70q tiny desktop", 48000, 62000, "nos"),
            ("Epson EB-X51 conference room projector", 38000, 48000, "nos"),
            ("Zebra DS2208 barcode scanner", 7500, 9500, "nos"),
            ("Anker USB-C 7-in-1 hub", 3000, 4500, "nos"),
            ("NVIDIA RTX 4000 Ada graphics card", 120000, 160000, "nos"),
            ("42U server rack cabinet", 45000, 70000, "nos"),
            ("Cat6 patch cords and RJ45 connectors", 3000, 9000, "lot"),
        ],
        "suppliers": ["Dell Technologies India Pvt Ltd", "HP India Sales Pvt Ltd", "Lenovo India Pvt Ltd",
                      "Redington Ltd", "Ingram Micro India", "Rashi Peripherals Ltd", "Apple India Pvt Ltd",
                      "Supertron Electronics"],
        "departments": ["IT", "IT", "Engineering", "Engineering", "Operations", "Finance"],
    },
    "Software & SaaS": {
        "description": "Software licences, subscriptions and SaaS tools (per-seat or annual), including renewals and support for software.",
        "items": [
            ("Microsoft 365 E3 licences annual", 2300, 2800, "seats"),
            ("Adobe Creative Cloud All Apps subscription", 6000, 7500, "seats"),
            ("Salesforce Sales Cloud Enterprise subscription", 12000, 15000, "seats"),
            ("Atlassian Jira Software Cloud Standard", 650, 900, "seats"),
            ("Confluence Cloud Premium", 900, 1200, "seats"),
            ("Slack Business+ plan", 1000, 1300, "seats"),
            ("Zoom Workplace Business licences", 1500, 2000, "seats"),
            ("Tableau Creator licence renewal", 5500, 6500, "seats"),
            ("Power BI Pro licences", 850, 1100, "seats"),
            ("Autodesk AutoCAD annual subscription", 140000, 170000, "seats"),
            ("MATLAB individual annual licence", 75000, 95000, "seats"),
            ("GitHub Enterprise Cloud seats", 1700, 2200, "seats"),
            ("Figma Organization plan", 3300, 4000, "seats"),
            ("Notion Plus workspace", 800, 1000, "seats"),
            ("Zoho Books Premium subscription", 3000, 4500, "seats"),
            ("Tally Prime Gold multi-user licence", 54000, 65000, "nos"),
            ("SAP S/4HANA professional user licences", 180000, 260000, "seats"),
            ("Oracle Database Enterprise Edition support renewal", 400000, 700000, "nos"),
            ("Windows Server 2022 Datacenter licence", 450000, 560000, "nos"),
            ("VMware vSphere Foundation subscription", 250000, 400000, "nos"),
            ("CrowdStrike Falcon endpoint protection", 4500, 6500, "seats"),
            ("Quick Heal Total Security antivirus", 1200, 1700, "seats"),
            ("Freshdesk Pro helpdesk subscription", 4000, 5000, "seats"),
            ("DocuSign Business Pro plan", 3200, 4000, "seats"),
            ("Canva for Teams subscription", 2500, 3200, "seats"),
            ("Grammarly Business licences", 1200, 1500, "seats"),
            ("JetBrains All Products Pack", 22000, 26000, "seats"),
            ("Postman Professional plan", 1000, 1300, "seats"),
            ("1Password Business", 600, 800, "seats"),
            ("Miro Business plan", 1300, 1700, "seats"),
        ],
        "suppliers": ["Microsoft Corporation India Pvt Ltd", "Adobe Systems Software Ireland", "Salesforce.com India",
                      "Atlassian Pty Ltd", "Softline Services India", "Crayon Software Experts India",
                      "Zoho Corporation", "Oracle India Pvt Ltd", "SoftwareOne India"],
        "departments": ["IT", "IT", "Engineering", "Marketing", "Finance", "Sales", "HR"],
    },
    "Cloud & Telecom Services": {
        "description": "Cloud infrastructure usage, internet / leased lines, mobile and telephony, hosting, managed IT and IT support services.",
        "items": [
            ("AWS cloud usage charges", 150000, 900000, "month"),
            ("Microsoft Azure consumption", 120000, 700000, "month"),
            ("Google Cloud Platform billing", 80000, 500000, "month"),
            ("Airtel 100 Mbps internet leased line", 35000, 60000, "month"),
            ("Jio Fiber business broadband", 3000, 8000, "month"),
            ("Vodafone Idea corporate postpaid plans", 400, 900, "connections"),
            ("Tata Communications MPLS link rental", 60000, 120000, "month"),
            ("Domain name renewal", 900, 2500, "nos"),
            ("SSL certificate wildcard renewal", 18000, 45000, "nos"),
            ("Data centre colocation rack rental", 90000, 180000, "month"),
            ("Managed IT support AMC", 60000, 150000, "month"),
            ("Network equipment AMC", 40000, 120000, "quarter"),
            ("IT helpdesk outsourcing services", 120000, 300000, "month"),
            ("Twilio SMS credits", 20000, 90000, "lot"),
            ("SendGrid email API credits", 8000, 30000, "month"),
            ("Toll-free number rental", 3000, 7000, "month"),
            ("Disaster recovery as a service", 70000, 160000, "month"),
            ("Laptop repair and servicing", 2500, 12000, "nos"),
            ("Cloudflare Enterprise CDN", 90000, 200000, "month"),
            ("Website hosting VPS", 2000, 9000, "month"),
            ("SD-WAN managed service", 50000, 110000, "month"),
            ("Exotel cloud telephony", 10000, 40000, "month"),
            ("Server migration professional services", 150000, 450000, "nos"),
            ("Penetration testing VAPT assessment", 150000, 400000, "nos"),
        ],
        "suppliers": ["Amazon Web Services India Pvt Ltd", "Microsoft Azure India", "Google Cloud India Pvt Ltd",
                      "Bharti Airtel Ltd", "Reliance Jio Infocomm Ltd", "Tata Communications Ltd",
                      "Vodafone Idea Ltd", "HCL Infosystems", "Wipro Ltd", "Sify Technologies"],
        "departments": ["IT", "IT", "IT", "Engineering", "Engineering", "Operations"],
    },
    "Office Supplies": {
        "description": "Stationery and consumables used at desks: paper, pens, files, toner and ink, small office items.",
        "items": [
            ("JK Copier A4 paper 75 GSM reams", 230, 320, "reams"),
            ("Reynolds ballpoint pens blue", 80, 120, "boxes"),
            ("Post-it sticky notes 3x3", 120, 200, "pads"),
            ("Kangaro stapler HD-45", 150, 260, "nos"),
            ("Staple pins No. 10", 15, 30, "boxes"),
            ("Plastic file folders", 25, 60, "nos"),
            ("Whiteboard markers assorted", 180, 300, "packs"),
            ("HP 88A toner cartridge", 4500, 6200, "nos"),
            ("Canon ink cartridge black", 1200, 2000, "nos"),
            ("Brown envelopes A4 size", 3, 8, "nos"),
            ("Classmate spiral notebooks", 60, 110, "nos"),
            ("Desk organiser tray", 300, 700, "nos"),
            ("Faber-Castell highlighters", 150, 250, "packs"),
            ("Casio DJ-120D calculator", 600, 900, "nos"),
            ("Office scissors 8 inch", 90, 180, "nos"),
            ("Fevistick glue stick", 25, 45, "nos"),
            ("Binder clips 32mm", 90, 150, "boxes"),
            ("ID card holders with lanyards", 25, 70, "nos"),
            ("Box files lever arch", 120, 220, "nos"),
            ("Correction pen", 30, 60, "nos"),
            ("A4 laminating pouches", 400, 700, "packs"),
            ("Printer paper A3", 600, 900, "reams"),
            ("Rubber stamps and stamp pad", 250, 600, "nos"),
            ("Whiteboard duster", 60, 150, "nos"),
        ],
        "suppliers": ["Kokuyo Camlin Ltd", "Navneet Education Ltd", "Sri Balaji Stationers", "Office Mart India",
                      "Rajesh Stationery Mart", "Hindustan Paper Traders"],
        "departments": ["Admin", "Admin", "Admin", "Finance", "HR", "Sales"],
    },
    "Facilities & Maintenance": {
        "description": "Running and maintaining office buildings: housekeeping, security, repairs, AMC for building equipment, furniture, office rent.",
        "items": [
            ("Housekeeping services monthly", 90000, 200000, "month"),
            ("Security guard services", 120000, 260000, "month"),
            ("Pest control treatment", 4000, 15000, "visit"),
            ("Split AC servicing AMC", 25000, 70000, "quarter"),
            ("Lizol floor cleaner and cleaning supplies", 2000, 8000, "lot"),
            ("Otis lift maintenance AMC", 40000, 90000, "quarter"),
            ("Electrical repair work", 5000, 40000, "job"),
            ("Plumbing repairs washroom", 3000, 25000, "job"),
            ("Godrej Interio workstations", 18000, 32000, "nos"),
            ("Featherlite ergonomic office chairs", 9000, 16000, "nos"),
            ("Office space rent", 600000, 1500000, "month"),
            ("Office painting work", 40000, 180000, "job"),
            ("CCTV camera installation", 30000, 120000, "job"),
            ("Fire extinguisher refilling", 800, 1500, "nos"),
            ("Aquaguard water purifier AMC", 3000, 6000, "nos"),
            ("Carpet shampooing and cleaning", 15000, 40000, "job"),
            ("Landscaping and garden maintenance", 15000, 40000, "month"),
            ("Waste management and disposal service", 10000, 30000, "month"),
            ("Office interior renovation", 500000, 2500000, "job"),
            ("Biometric access control system", 40000, 120000, "nos"),
            ("Conference room table", 35000, 90000, "nos"),
            ("Tissue rolls and hand wash refills", 3000, 9000, "lot"),
            ("DG set maintenance service", 15000, 45000, "quarter"),
            ("Building maintenance charges CAM", 150000, 400000, "month"),
        ],
        "suppliers": ["Godrej Interio", "ISS Facility Services India", "JLL Property Services", "CBRE South Asia",
                      "Otis Elevator Company India", "Blue Star Ltd", "Rentokil PCI Pest Control",
                      "Shree Ganesh Facility Services", "Featherlite Furniture"],
        "departments": ["Admin", "Admin", "Admin", "Operations", "Facilities", "Facilities"],
    },
    "Professional Services": {
        "description": "External expert services: audit, legal, tax, consulting, certification and advisory fees.",
        "items": [
            ("Statutory audit fees", 400000, 1500000, "engagement"),
            ("Legal retainer fees", 150000, 400000, "month"),
            ("Contract review legal opinion", 50000, 250000, "nos"),
            ("GST compliance and return filing", 20000, 60000, "month"),
            ("Management consulting engagement", 1500000, 6000000, "engagement"),
            ("Transfer pricing study", 300000, 900000, "engagement"),
            ("Trademark registration filing", 15000, 60000, "nos"),
            ("Financial due diligence", 800000, 3000000, "engagement"),
            ("Actuarial valuation gratuity", 60000, 150000, "engagement"),
            ("IT strategy advisory", 900000, 3500000, "engagement"),
            ("Company secretarial services", 25000, 70000, "month"),
            ("Business valuation report", 150000, 500000, "nos"),
            ("ISO 27001 certification audit", 250000, 600000, "engagement"),
            ("Internal audit services", 200000, 700000, "quarter"),
            ("Tax advisory income tax assessment", 100000, 400000, "engagement"),
            ("Architect consultancy fees", 200000, 900000, "engagement"),
            ("Patent drafting and filing", 80000, 250000, "nos"),
            ("Notary and stamp duty charges", 1000, 10000, "nos"),
            ("Arbitration legal representation", 500000, 2000000, "engagement"),
            ("Forensic audit investigation", 700000, 2500000, "engagement"),
        ],
        "suppliers": ["Deloitte Haskins & Sells LLP", "KPMG Assurance and Consulting", "Ernst & Young LLP",
                      "PricewaterhouseCoopers Pvt Ltd", "Grant Thornton Bharat LLP", "Cyril Amarchand Mangaldas",
                      "Khaitan & Co", "AZB & Partners", "McKinsey & Company India", "Boston Consulting Group India"],
        "departments": ["Finance", "Finance", "Legal", "Legal", "Management", "Compliance"],
    },
    "Marketing & Advertising": {
        "description": "Promoting the company: digital ads, media buying, agencies, events, printed collateral, branded merchandise.",
        "items": [
            ("Google Ads search campaign spend", 100000, 800000, "month"),
            ("Meta Facebook Instagram ads", 80000, 600000, "month"),
            ("LinkedIn sponsored content campaign", 100000, 500000, "campaign"),
            ("Print advertisement Times of India", 200000, 900000, "insertion"),
            ("FM radio spots", 50000, 250000, "campaign"),
            ("Hoarding billboard rental", 100000, 500000, "month"),
            ("Influencer marketing campaign", 100000, 700000, "campaign"),
            ("Event sponsorship fee", 300000, 2000000, "event"),
            ("Trade show booth design and fabrication", 250000, 900000, "event"),
            ("Branded t-shirts and coffee mugs", 250, 600, "nos"),
            ("Product brochures printing", 15, 60, "nos"),
            ("SEO agency retainer", 60000, 200000, "month"),
            ("PR agency retainer", 150000, 500000, "month"),
            ("Corporate video production", 300000, 1200000, "nos"),
            ("Product photography shoot", 50000, 200000, "job"),
            ("Content writing services blogs", 3000, 8000, "articles"),
            ("Roll-up standees and banners", 1500, 4000, "nos"),
            ("Visiting cards printing", 300, 900, "boxes"),
            ("Product launch event management", 500000, 2500000, "event"),
            ("Market research survey study", 300000, 1200000, "engagement"),
            ("YouTube pre-roll video ads", 100000, 600000, "campaign"),
            ("Mailchimp email campaign credits", 15000, 50000, "month"),
        ],
        "suppliers": ["Google India Pvt Ltd", "Meta Platforms Ireland Ltd", "LinkedIn Ireland Unlimited",
                      "Bennett Coleman & Co Ltd", "Ogilvy & Mather India", "Dentsu India", "Madison Media",
                      "Vistaprint India", "Times OOH Media", "Wizcraft International"],
        "departments": ["Marketing", "Marketing", "Marketing", "Sales", "Sales", "Management"],
    },
    "Travel & Entertainment": {
        "description": "Employee business travel and client entertainment: flights, hotels, cabs, trains, visas, meals with clients, offsites.",
        "items": [
            ("Flight ticket BLR-DEL return", 9000, 22000, "tickets"),
            ("International flight BOM-SIN", 35000, 75000, "tickets"),
            ("Hotel stay Taj 2 nights", 14000, 35000, "nights"),
            ("Marriott hotel accommodation", 9000, 20000, "nights"),
            ("Uber for Business rides", 3000, 25000, "month"),
            ("Ola corporate cab bookings", 3000, 20000, "month"),
            ("IRCTC train tickets 2AC", 1500, 4500, "tickets"),
            ("US visa application fees", 15000, 17000, "nos"),
            ("Travel insurance overseas trip", 1500, 5000, "nos"),
            ("Client dinner", 6000, 30000, "nos"),
            ("Team offsite resort booking", 200000, 900000, "event"),
            ("Airport pickup and drop", 1200, 3000, "trips"),
            ("Self-drive car rental Zoomcar", 3000, 9000, "days"),
            ("Travel agency service fee", 300, 900, "bookings"),
            ("Forex card loading for travel", 50000, 200000, "nos"),
            ("Hotel conference hall booking", 60000, 250000, "event"),
            ("Business lounge access", 2000, 4000, "nos"),
            ("Outstation taxi Pune trip", 4000, 12000, "trips"),
            ("Client entertainment golf event", 50000, 200000, "event"),
            ("Daily allowance reimbursement travel", 1500, 5000, "days"),
        ],
        "suppliers": ["MakeMyTrip India Pvt Ltd", "Yatra Online Ltd", "Cleartrip Pvt Ltd",
                      "InterGlobe Aviation Ltd", "Air India Ltd", "Uber India Systems Pvt Ltd",
                      "ANI Technologies Pvt Ltd", "Indian Hotels Company Ltd", "Marriott Hotels India",
                      "IRCTC", "Thomas Cook India"],
        "departments": ["Sales", "Sales", "Management", "Engineering", "Marketing", "Operations", "HR"],
    },
    "Logistics & Freight": {
        "description": "Moving goods: couriers, domestic and international freight, customs clearance, warehousing and 3PL.",
        "items": [
            ("Blue Dart express courier charges", 5000, 40000, "month"),
            ("DTDC domestic courier", 3000, 25000, "month"),
            ("FedEx international express shipment", 6000, 60000, "shipments"),
            ("Ocean freight FCL Nhava Sheva to Rotterdam", 150000, 400000, "containers"),
            ("Air freight charges export", 80000, 500000, "shipments"),
            ("Customs clearance CHA charges", 8000, 35000, "shipments"),
            ("Warehouse storage rent 3PL", 150000, 600000, "month"),
            ("FTL trucking Pune to Chennai", 45000, 90000, "trips"),
            ("Last mile delivery charges", 30, 90, "parcels"),
            ("Freight forwarding services", 40000, 200000, "shipments"),
            ("Container demurrage charges", 20000, 150000, "nos"),
            ("Port handling and THC charges", 10000, 40000, "containers"),
            ("Cold chain refrigerated transport", 30000, 120000, "trips"),
            ("Delhivery B2B shipments", 20000, 150000, "month"),
            ("Shiprocket shipping charges", 10000, 80000, "month"),
            ("Inbound freight on raw material", 15000, 80000, "trips"),
            ("Loading unloading labour charges", 3000, 15000, "trips"),
            ("Transit insurance marine cargo", 10000, 50000, "nos"),
            ("Pallet storage and handling", 20000, 80000, "month"),
            ("Rail freight container", 60000, 150000, "containers"),
        ],
        "suppliers": ["Blue Dart Express Ltd", "DTDC Express Ltd", "FedEx Express TSCS India",
                      "DHL Express India", "Delhivery Ltd", "Maersk Line India", "Gati Ltd",
                      "VRL Logistics Ltd", "TCI Freight", "Allcargo Logistics Ltd", "Container Corporation of India"],
        "departments": ["Supply Chain", "Supply Chain", "Supply Chain", "Operations", "Operations", "Sales"],
    },
    "Raw Materials": {
        "description": "Direct materials that go into manufactured products: metals, polymers, chemicals, yarn, components.",
        "items": [
            ("Hot rolled steel coils", 52000, 65000, "MT"),
            ("Aluminium ingots", 210000, 250000, "MT"),
            ("Copper wire rods 8mm", 720000, 820000, "MT"),
            ("Polypropylene granules", 95000, 120000, "MT"),
            ("HDPE resin", 100000, 125000, "MT"),
            ("Cotton yarn 30s count", 250, 330, "kg"),
            ("OPC 53 grade cement bags", 350, 450, "bags"),
            ("TMT bars Fe 500D", 55000, 68000, "MT"),
            ("ABS plastic pellets", 140000, 180000, "MT"),
            ("Rubber compound", 180, 280, "kg"),
            ("Float glass sheets 5mm", 600, 1100, "sheets"),
            ("Caustic soda flakes", 40, 60, "kg"),
            ("Sulphuric acid 98%", 8, 15, "kg"),
            ("Titanium dioxide pigment", 300, 420, "kg"),
            ("Epoxy resin", 350, 500, "kg"),
            ("FR4 PCB laminates", 1500, 3000, "sheets"),
            ("SMD resistors and capacitors reels", 800, 3000, "reels"),
            ("Isopropyl alcohol solvent", 120, 180, "litres"),
            ("Stainless steel sheets 304", 220, 300, "kg"),
            ("Plywood boards 18mm", 2200, 3500, "sheets"),
            ("Brass rods", 520, 650, "kg"),
            ("Lithium ion battery cells 18650", 180, 320, "nos"),
        ],
        "suppliers": ["Tata Steel Ltd", "JSW Steel Ltd", "Hindalco Industries Ltd", "Vedanta Ltd",
                      "Reliance Industries Ltd (Polymers)", "UltraTech Cement Ltd", "Grasim Industries",
                      "Haldia Petrochemicals Ltd", "Vardhman Textiles", "Arrow Electronics India"],
        "departments": ["Production", "Production", "Production", "Procurement", "Procurement", "R&D"],
    },
    "MRO & Industrial Supplies": {
        "description": "Maintenance, repair and operations items for factories and plants: safety gear, tools, spares, lubricants, fasteners.",
        "items": [
            ("Safety shoes steel toe", 900, 2200, "pairs"),
            ("Safety helmets ISI marked", 150, 400, "nos"),
            ("Nitrile disposable gloves", 400, 800, "boxes"),
            ("SKF 6205 ball bearings", 250, 600, "nos"),
            ("V-belts B-section", 200, 500, "nos"),
            ("Hydraulic oil 68 grade", 180, 260, "litres"),
            ("Servo lubricant grease", 250, 450, "kg"),
            ("Welding electrodes E6013", 300, 600, "packets"),
            ("Bosch grinding and cutting discs", 40, 120, "nos"),
            ("Taparia hand tools kit", 2500, 6000, "sets"),
            ("Bosch GSB 500W impact drill", 3000, 4500, "nos"),
            ("PPE kits coveralls", 300, 700, "nos"),
            ("3 phase induction motor 5HP spare", 22000, 35000, "nos"),
            ("Pneumatic fittings and connectors", 50, 300, "nos"),
            ("Industrial hose pipes", 2000, 6000, "rolls"),
            ("Hex bolts nuts and washers", 3, 20, "nos"),
            ("Spray paint cans", 150, 300, "nos"),
            ("Fevicol industrial adhesive", 300, 700, "kg"),
            ("CNC machine spare parts", 10000, 80000, "nos"),
            ("Pump mechanical seal repair kit", 4000, 15000, "nos"),
            ("Nylon cable ties", 100, 300, "packets"),
            ("Fluke digital multimeter", 9000, 18000, "nos"),
            ("Aluminium folding ladder 8 ft", 4000, 8000, "nos"),
            ("Safety goggles", 80, 250, "nos"),
            ("Ear plugs and ear muffs", 30, 600, "nos"),
        ],
        "suppliers": ["Industrybuying.com", "Moglix (Mogli Labs India)", "SKF India Ltd", "Bosch Ltd",
                      "Taparia Tools Ltd", "Karam Safety Pvt Ltd", "3M India Ltd", "Castrol India Ltd",
                      "Indian Oil Corporation Ltd", "Grundfos Pumps India"],
        "departments": ["Maintenance", "Maintenance", "Production", "Production", "EHS", "Operations"],
    },
    "Utilities & Energy": {
        "description": "Electricity, water, gas and fuel bills for sites; diesel for generators; power purchase agreements.",
        "items": [
            ("Electricity bill HT connection", 200000, 1200000, "month"),
            ("Electricity charges office LT", 40000, 180000, "month"),
            ("Water charges and sewerage", 8000, 40000, "month"),
            ("High speed diesel for DG set", 88, 96, "litres"),
            ("Piped natural gas PNG bill", 30000, 200000, "month"),
            ("Commercial LPG cylinders 19kg", 1700, 2100, "cylinders"),
            ("Solar power PPA charges", 100000, 500000, "month"),
            ("CNG for company vehicles", 75, 90, "kg"),
            ("Water tanker supply", 1500, 4000, "tankers"),
            ("Open access power charges", 300000, 1000000, "month"),
            ("Petrol for pool vehicles", 100, 108, "litres"),
            ("Electricity meter deposit", 20000, 100000, "nos"),
            ("Steam supply charges", 100000, 400000, "month"),
            ("Furnace oil", 55, 70, "litres"),
        ],
        "suppliers": ["BESCOM", "MSEDCL", "Tata Power Company Ltd", "Adani Electricity Mumbai Ltd", "BWSSB",
                      "Indian Oil Corporation Ltd", "Bharat Petroleum Corporation Ltd",
                      "Hindustan Petroleum Corporation Ltd", "Mahanagar Gas Ltd", "Indraprastha Gas Ltd"],
        "departments": ["Facilities", "Facilities", "Admin", "Operations", "Production"],
    },
    "HR & Training": {
        "description": "People-related spend: recruitment, staffing, background checks, learning and training, employee benefits and insurance.",
        "items": [
            ("Recruitment agency placement fee", 80000, 400000, "hires"),
            ("Naukri job postings and resume access", 60000, 250000, "nos"),
            ("LinkedIn Recruiter Lite seats", 9000, 14000, "seats"),
            ("Employee background verification", 1200, 3500, "checks"),
            ("Leadership development workshop", 150000, 700000, "programme"),
            ("Coursera for Business licences", 25000, 33000, "seats"),
            ("Udemy Business subscription", 30000, 36000, "seats"),
            ("Executive coaching programme", 200000, 800000, "programme"),
            ("Campus hiring drive expenses", 50000, 250000, "event"),
            ("Contract staffing temporary staff", 30000, 60000, "headcount"),
            ("Payroll processing outsourcing", 40000, 120000, "month"),
            ("Group health insurance premium", 12000, 25000, "employees"),
            ("Annual employee health checkups", 2000, 5000, "employees"),
            ("Team building activity", 50000, 200000, "event"),
            ("Employee rewards Amazon gift vouchers", 1000, 5000, "vouchers"),
            ("Relocation services new joiners", 40000, 150000, "nos"),
            ("Technical conference registration fee", 15000, 60000, "nos"),
            ("POSH awareness training", 30000, 100000, "programme"),
            ("Employee assistance programme", 50000, 150000, "quarter"),
            ("Psychometric assessment tests", 1000, 3000, "candidates"),
        ],
        "suppliers": ["Info Edge India Ltd (Naukri)", "TeamLease Services Ltd", "Quess Corp Ltd", "Randstad India",
                      "AuthBridge Research Services", "First Advantage India", "Coursera Inc", "Udemy Inc",
                      "ICICI Lombard General Insurance", "Star Health Insurance", "ADP India"],
        "departments": ["HR", "HR", "HR", "HR", "Management", "Engineering"],
    },
    "Food & Catering": {
        "description": "Food and beverages for employees and events: cafeteria, pantry supplies, meals, snacks, drinking water.",
        "items": [
            ("Cafeteria catering services", 200000, 700000, "month"),
            ("Pantry tea and coffee supplies", 5000, 20000, "month"),
            ("Milk supply for pantry", 60, 70, "litres"),
            ("Bisleri 20L water cans", 70, 110, "cans"),
            ("Snacks and biscuits for pantry", 4000, 15000, "month"),
            ("Team lunch order", 5000, 30000, "nos"),
            ("Working dinner food delivery", 3000, 15000, "nos"),
            ("Diwali sweets boxes", 400, 900, "boxes"),
            ("Event catering buffet", 600, 1500, "plates"),
            ("Nescafe coffee vending machine rental", 4000, 9000, "month"),
            ("Fresh fruit basket delivery", 2000, 6000, "week"),
            ("Office breakfast service", 50000, 150000, "month"),
            ("Packaged drinking water bottles 500ml", 8, 14, "bottles"),
            ("Birthday cakes celebration", 800, 3000, "nos"),
            ("Chai and samosa for meeting", 500, 2000, "nos"),
            ("Canteen food subsidy", 100000, 400000, "month"),
        ],
        "suppliers": ["Sodexo Food Solutions India", "Compass Group India", "Bundl Technologies (Swiggy)",
                      "Zomato Ltd", "Haldiram Snacks Pvt Ltd", "Bisleri International", "Nestle Professional",
                      "Chaayos (Sunshine Teahouse)", "Annapurna Caterers", "Elior India Food Services"],
        "departments": ["Admin", "Admin", "HR", "HR", "Facilities", "Management"],
    },
    "Packaging Materials": {
        "description": "Materials used to pack and ship products: boxes, cartons, films, tapes, pallets, labels, containers.",
        "items": [
            ("Corrugated boxes 5 ply", 25, 80, "nos"),
            ("Bubble wrap rolls", 600, 1500, "rolls"),
            ("Stretch wrap film 23 micron", 150, 220, "kg"),
            ("BOPP packing tape 48mm", 30, 60, "rolls"),
            ("Wooden pallets heat treated", 700, 1400, "nos"),
            ("EPE foam inserts", 10, 60, "nos"),
            ("Product labels printing barcode stickers", 1, 4, "nos"),
            ("Shrink wrap film", 160, 240, "kg"),
            ("LDPE poly bags", 120, 200, "kg"),
            ("PP strapping rolls", 900, 1500, "rolls"),
            ("Kraft paper void fill", 60, 110, "kg"),
            ("Tin containers with lids", 15, 45, "nos"),
            ("Amber glass bottles 100ml", 6, 14, "nos"),
            ("Laminated pouches printed", 2, 6, "nos"),
            ("Mono cartons printed", 4, 15, "nos"),
            ("Edge protectors corner boards", 8, 20, "nos"),
            ("PET jars 500ml", 5, 12, "nos"),
            ("Jumbo bags FIBC", 350, 700, "nos"),
        ],
        "suppliers": ["TCPL Packaging Ltd", "Uflex Ltd", "EPL Ltd (Essel Propack)", "Huhtamaki India Ltd",
                      "Packman Packaging Pvt Ltd", "Avery Dennison India", "Sai Corrugated Boxes",
                      "Shree Balaji Packaging Industries"],
        "departments": ["Supply Chain", "Supply Chain", "Production", "Production", "Operations"],
    },
}

# Marketplaces / distributors that sell almost anything -> supplier name gives no hint.
GENERIC_SUPPLIERS = ["Amazon Business (Amazon Seller Services)", "Flipkart Wholesale", "IndiaMART Supplier",
                     "Reliance Retail Ltd", "Tata Enterprises", "Local Vendor - Cash Purchase",
                     "Sharma Enterprises", "Om Sai Traders", "Global Traders & Co", "Misc Vendor"]

GENERIC_WEIGHTS = [10, 4, 5, 3, 2, 2, 2, 2, 2, 1]

# Real suppliers often sell into several categories (distributors, big conglomerates).
SHARED_SUPPLIERS = {
    "Software & SaaS": ["Redington Ltd", "Ingram Micro India", "Google India Pvt Ltd", "Wipro Ltd"],
    "Cloud & Telecom Services": ["Redington Ltd", "Google India Pvt Ltd", "Microsoft Corporation India Pvt Ltd"],
    "IT Hardware": ["Croma (Infiniti Retail)", "Reliance Digital"],
    "Office Supplies": ["3M India Ltd", "Avery Dennison India", "Croma (Infiniti Retail)"],
    "Facilities & Maintenance": ["Sodexo Facilities Management", "Blue Star Ltd", "Bosch Ltd"],
    "Professional Services": ["Wipro Ltd", "Ernst & Young LLP"],
    "Marketing & Advertising": ["Avery Dennison India", "Indian Hotels Company Ltd"],
    "Travel & Entertainment": ["Bundl Technologies (Swiggy)", "Zomato Ltd"],
    "Logistics & Freight": ["Reliance Industries Ltd (Polymers)", "Tata Steel Ltd"],
    "HR & Training": ["LinkedIn Ireland Unlimited", "Wizcraft International", "Indian Hotels Company Ltd"],
    "Food & Catering": ["Indian Hotels Company Ltd", "Marriott Hotels India", "Sodexo Facilities Management"],
    "MRO & Industrial Supplies": ["Bharat Petroleum Corporation Ltd", "Hindustan Petroleum Corporation Ltd"],
    "Packaging Materials": ["Reliance Industries Ltd (Polymers)", "Uflex Ltd", "3M India Ltd"],
    "Utilities & Energy": ["Tata Power Company Ltd"],
}
for _cat, _sups in SHARED_SUPPLIERS.items():
    CATALOGUE[_cat]["suppliers"] = CATALOGUE[_cat]["suppliers"] + _sups

# Short, almost content-free descriptions (classic "lazy" AP entries)
VAGUE_DESCRIPTIONS = ["Inv {inv} - charges for {mon}", "Payment against PO {po}", "Bill no {inv}",
                      "Monthly charges {mon}", "Services rendered {mon}", "Material supply as per PO {po}",
                      "Reimbursement claim {inv}", "Advance payment", "As per quotation {inv}"]

ABBREVIATIONS = {
    "licence": "lic", "licences": "lic", "subscription": "subscr", "services": "svcs", "service": "svc",
    "maintenance": "maint", "charges": "chgs", "monthly": "mthly", "annual": "annl", "international": "intl",
    "management": "mgmt", "professional": "prof", "renewal": "renwl", "printing": "prntg", "and": "&",
    "with": "w/", "for": "4", "quantity": "qty", "assorted": "asstd", "equipment": "eqpt",
    "employee": "emp", "insurance": "ins", "transport": "transpt", "delivery": "dlvry",
}

TEMPLATES = ["{item}", "{item}", "{item} - {qty} {unit}", "Purchase of {item}", "Supply of {item}",
             "{item} for {dept} team", "{item} ({mon})", "{item} x{qty}", "Being {item} purchased",
             "{item} | qty {qty}", "{qty} {unit} {item}", "Procurement - {item}", "{item} as per PO {po}"]
RECURRING_TEMPLATES = ["{item} for {mon}", "{item} - {mon} invoice", "Renewal: {item}",
                       "{item} period {mon}", "{item} chgs {mon}"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
RECURRING_UNITS = {"month", "quarter", "week", "seats"}


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 5:
        return word
    i = rng.randrange(1, len(word) - 2)
    op = rng.choice(["swap", "drop", "double"])
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop":
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]


def add_noise(text: str, rng: random.Random) -> str:
    words = text.split()
    out = []
    for w in words:
        lw = w.lower()
        if lw in ABBREVIATIONS and rng.random() < 0.35:
            w = ABBREVIATIONS[lw]
        elif rng.random() < 0.04:
            w = _typo(w, rng)
        out.append(w)
    # randomly drop one non-first word
    if len(out) > 4 and rng.random() < 0.15:
        out.pop(rng.randrange(1, len(out)))
    text = " ".join(out)
    r = rng.random()
    if r < 0.25:
        text = text.upper()
    elif r < 0.35:
        text = text.lower()
    # codes prepended / appended like real ERP exports
    r = rng.random()
    if r < 0.15:
        text = f"PO-{rng.randint(4500000000, 4599999999)} {text}"
    elif r < 0.25:
        text = f"{text} /INV{rng.randint(1000, 99999)}"
    elif r < 0.30:
        text = f"SKU-{rng.randint(10000, 99999)} {text}"
    elif r < 0.34:
        text = f"{text} GRN#{rng.randint(100000, 999999)}"
    if rng.random() < 0.08:  # truncated by a fixed-width ERP field
        text = text[: rng.randint(18, 32)]
    return text.strip()


def random_date(rng: random.Random, start=date(2025, 4, 1), end=date(2026, 3, 31)) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))


def generate(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    categories = list(CATALOGUE)
    # some categories are bought more often than others
    weights = [10, 9, 8, 7, 8, 5, 7, 9, 6, 6, 7, 4, 5, 5, 4]
    rows = []
    for i in range(n):
        cat = rng.choices(categories, weights=weights)[0]
        spec = CATALOGUE[cat]
        item, lo, hi, unit = rng.choice(spec["items"])
        recurring = unit in RECURRING_UNITS
        if unit in {"nos", "seats", "boxes", "packs", "reams", "pairs", "kg", "litres", "MT", "bags",
                    "sheets", "rolls", "cans", "bottles", "parcels", "employees", "vouchers", "plates",
                    "tickets", "nights", "checks", "connections", "cylinders", "packets", "sets",
                    "reels", "candidates", "articles", "headcount", "shipments", "containers", "trips",
                    "days", "bookings", "hires", "tankers", "lot"}:
            qty_ranges = {"nos": (1, 40), "seats": (5, 300), "MT": (1, 60), "kg": (20, 3000),
                          "litres": (100, 8000), "parcels": (100, 5000), "bottles": (200, 3000),
                          "plates": (40, 400), "employees": (20, 500), "lot": (1, 3)}
            qa, qb = qty_ranges.get(unit, (1, 200))
            if hi >= 50000:  # expensive things are bought in small numbers
                qa, qb = 1, min(qb, 8)
            elif hi >= 10000:
                qa, qb = qa, min(qb, 40)
            qty = rng.randint(qa, qb)
        else:
            qty = 1
        amount = round(qty * rng.uniform(lo, hi), 2)
        d = random_date(rng)
        dept = rng.choice(spec["departments"]) if rng.random() < 0.85 else rng.choice(
            ["Admin", "Finance", "IT", "Operations", "Sales", "Marketing", "HR"])
        mon = f"{MONTHS[d.month - 1]}-{str(d.year)[2:]}"
        po = rng.randint(4500000000, 4599999999)
        inv = rng.randint(1000, 99999)

        # marketplaces / local traders mostly supply cheaper items
        p_generic = 0.55 if hi < 20000 else 0.30 if hi < 200000 else 0.12
        if rng.random() < p_generic:
            supplier = rng.choices(GENERIC_SUPPLIERS, weights=GENERIC_WEIGHTS)[0]
        else:
            supplier = rng.choice(spec["suppliers"])
        if rng.random() < 0.07:  # lazy entry: description says nothing useful
            desc = rng.choice(VAGUE_DESCRIPTIONS).format(inv=inv, mon=mon, po=po)
        else:
            tpl = rng.choice(RECURRING_TEMPLATES if recurring and rng.random() < 0.5 else TEMPLATES)
            desc = tpl.format(item=item, qty=qty, unit=unit, dept=dept, mon=mon, po=po)
            desc = add_noise(desc, rng)

        rows.append({
            "transaction_id": f"TXN{100000 + i}",
            "invoice_date": d.isoformat(),
            "supplier_name": supplier,
            "line_description": desc,
            "department": dept,
            "amount_inr": amount,
            "category": cat,
            "_item": item,  # kept only for the train/test split (unseen-item evaluation)
        })
    df = pd.DataFrame(rows).sort_values("invoice_date").reset_index(drop=True)
    return df


def write_taxonomy():
    taxonomy = [{"category": c, "description": v["description"]} for c, v in CATALOGUE.items()]
    config.TAXONOMY_PATH.write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")
    return taxonomy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=5000, help="number of procurement records")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    write_taxonomy()
    df = generate(args.n, args.seed)
    df.to_csv(config.RAW_DATA_PATH, index=False)
    print(f"Wrote {len(df)} records -> {config.RAW_DATA_PATH}")
    print(df["category"].value_counts().to_string())
    print("\nSample rows:")
    print(df[["supplier_name", "line_description", "category"]].sample(8, random_state=1).to_string())


if __name__ == "__main__":
    main()
