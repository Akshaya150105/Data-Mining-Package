"""
district_mapping.py
====================
Manual mapping of DB district names → 2011 shapefile district names.
Covers all 137 unmatched districts:
  - Political renames (Gurugram, Ayodhya, Mysuru...)
  - Spelling variants (Ananthapuramu, Howrah...)
  - Post-2011 new districts mapped to their parent district
  - Abbreviated names (Y S R, N. T. R)
 
Usage: import DISTRICT_MAP and use in clustering.py match_shp()
"""
 
DISTRICT_MAP = {
 
    # ── Political / official renames ───────────────────────────────────────
    'Mysuru':                       'Mysore',
    'Belagavi':                     'Belgaum',
    'Kalaburagi':                   'Gulbarga',
    'Ballari':                      'Bellary',
    'Vijayapura':                   'Bijapur',
    'Vijayanagara':                 'Bellary',        # carved from Bellary 2021
    'Bengaluru Urban':              'Bangalore',
    'Bengaluru Rural':              'Bangalore Rural',
    'Gurugram':                     'Gurgaon',
    'Ayodhya':                      'Faizabad',
    'Howrah':                       'Haora',
    'Hooghly':                      'Hugli',
    'Amroha':                       'Jyotiba Phule Nagar',
    'Hathras':                      'Mahamaya Nagar',
    'Kasganj':                      'Kanshiram Nagar',
    'Shamli':                       'Prabudh Nagar',
    'Amethi':                       'Chhatrapati Sahuji Mahraj Nagar',
    'Ananthapuramu':                'Anantapur',
    'Eluru':                        'West Godavari',  # carved 2022, use parent
    'Kakinada':                     'East Godavari',
    'Tirupati':                     'Chittoor',
    'Bapatla':                      'Guntur',         # carved from Guntur 2022
    'Anakapalli':                   'Visakhapatnam',  # carved 2022
    'Nandyal':                      'Kurnool',        # carved 2022
    'Palnadu':                      'Guntur',         # carved 2022
    'Annamayya':                    'Kadapa',         # carved 2022
    'Alluri Sitharama Raju':        'East Godavari',
    'Parvathipuram Manyam':         'Vizianagaram',
    'Y S R':                        'Kadapa',
    'N. T. R':                      'Krishna',
    'Shopian':                      'Shupiyan',
    'Chhatrapati Sambhajinagar':    'Aurangabad',
    'Beed':                         'Bid',
    'Raigad':                       'Raigarh',        # note: Maharashtra Raigad
    'Palghar':                      'Thane',          # carved from Thane 2014
    'Dahod':                        'Dohad',
    'Gir Somnath':                  'Junagadh',       # carved 2013
    'Botad':                        'Bhavnagar',      # carved 2013
    'Morbi':                        'Rajkot',         # carved 2013
    'Devbhumi Dwarka':              'Jamnagar',       # carved 2013
    'Arvalli':                      'Sabarkantha',    # carved 2013
    'Chhotaudepur':                 'Vadodara',       # carved 2013
    'Khargone':                     'West Nimar',
    'Khandwa':                      'East Nimar',
    'Narsinghpur':                  'Narsimhapur',
    'Mungeli':                      'Bilaspur',       # carved 2012
    'Baloda Bazar':                 'Raipur',         # carved 2012
    'Balod':                        'Durg',           # carved 2012
    'Bemetara':                     'Durg',           # carved 2012
    'Gariyaband':                   'Raipur',         # carved 2012
    'Surajpur':                     'Surguja',        # carved 2012
    'Sakti':                        'Janjgir-Champa', # carved 2022
    'Kondagaon':                    'Bastar',         # carved 2012
    'Gaurela-Pendra-Marwahi':       'Bilaspur',
    'Khairagarh-Chhuikhadan-Gandai':'Rajnandgaon',
    'Mohla-Manpur-Ambagarh Chouki': 'Rajnandgaon',
    'Maihar':                       'Satna',          # carved 2020
    'Pandhurna':                    'Chhindwara',     # carved 2023
    'Niwari':                       'Tikamgarh',      # carved 2018
    'Deeg':                         'Bharatpur',      # carved 2023
    'Beawar':                       'Ajmer',          # carved 2023
    'Phalodi':                      'Jodhpur',        # carved 2023
    'Salumbar':                     'Udaipur',        # carved 2023
    'Didwana-Kuchaman':             'Nagaur',         # carved 2023
    'Khairthal-Tijara':             'Alwar',          # carved 2023
    'Kotputli-Behror':              'Jaipur',         # carved 2023
    'Balotra':                      'Barmer',         # carved 2023
    'Malerkotla':                   'Sangrur',        # carved 2021
    'Pathankot':                    'Gurdaspur',
    'Mohali':                       'Sahibzada Ajit Singh Nagar',
    'Fazilka':                      'Firozpur',       # carved 2011
    'Nuh':                          'Mewat',
    'Charkhi Dadri':                'Bhiwani',        # carved 2016
    'Shahdara':                     'East Delhi',     # carved from East Delhi
    'Jhargram':                     'Paschim Medinipur',  # carved 2017
    'Kalimpong':                    'Darjiling',      # carved 2017
    'Alipurduar':                   'Jalpaiguri',     # carved 2014
    'Cooch Behar':                  'Koch Bihar',
    'Sribhumi':                     'Birbhum',        # renamed
    'Bajali':                       'Barpeta',        # carved 2021
    'Biswanath':                    'Sonitpur',       # carved 2015
    'Hojai':                        'Nagaon',         # carved 2015
    'Majuli':                       'Jorhat',         # carved 2016
    'Charaideo':                    'Sibsagar',       # carved 2015
    'Gomati':                       'Tripura East',   # renamed
    'Khowai':                       'Tripura West',
    'Sepahijala':                   'Tripura West',   # carved 2012
    'Tenkasi':                      'Tirunelveli',    # carved 2019
    'Tirupattur':                   'Vellore',        # carved 2019
    'Ranipet':                      'Vellore',        # carved 2019
    'Chengalpattu':                 'Kanchipuram',    # carved 2019
    'Kallakurichi':                 'Villupuram',     # carved 2019
    'Mayiladuthurai':               'Nagapattinam',   # carved 2021
    'Namchi':                       'South Sikkim',
    'Longding':                     'Tirap',          # carved 2012
    'Kakching':                     'Thoubal',        # carved 2016
    'Jiribam':                      'Imphal East',    # carved 2016
    'Nandyal':                      'Kurnool',
    'Jagitial':                     'Karimnagar',     # carved 2016
    'Rajanna Sircilla':             'Karimnagar',
    'Peddapalli':                   'Karimnagar',
    'Siddipet':                     'Medak',
    'Mancherial':                   'Adilabad',
    'Nirmal':                       'Adilabad',
    'Komaram Bheem':                'Adilabad',
    'Vikarabad':                    'Ranga Reddy',
    'Wanaparthy':                   'Mahabubnagar',
    'Narayanpet':                   'Mahabubnagar',
    'Jogulamba Gadwal':             'Mahabubnagar',
    'Kamareddy':                    'Nizamabad',
    'Mahabubabad':                  'Warangal',
    'Hanumakonda':                  'Warangal',
    'Mulugu':                       'Warangal',
    'Jayashankar Bhupalapally':     'Karimnagar',
    'Medchal-Malkajgiri':           'Ranga Reddy',
    'Suryapet':                     'Nalgonda',
    'Yadadri-Bhuvanagiri':          'Nalgonda',
    'Boudh':                        'Boudh',
    'Sukma':                        'Dantewada',
    'Leparada':                     'East Siang',
    'Pakke Kessang':                'East Kameng',
    'Shi-Yomi':                     'West Siang',
    'Kra Daadi':                    'Kurung Kumey',
    'Kamle':                        'Upper Subansiri',
    'Namsai':                       'Lohit',
    'Noklak':                       'Tuensang',
    'Tseminyu':                     'Kohima',
    'Chumoukedima':                 'Dimapur',
    'Shamator':                     'Tuensang',
    'Meluri':                       'Phek',
    'Niuland':                      'Dimapur',
    'Hnahthial':                    'Lunglei',
    'Khawzawl':                     'Champhai',
    'Saitual':                      'Aizawl',
    'Pherzawl':                     'Churachandpur',
    'Tamulpur':                     'Baksa',
    'Tiswadi':                      'North Goa',
    'Bardez':                       'North Goa',
 
}
 
def apply_mapping(db_name: str) -> str:
    """
    Returns the shapefile name for a DB district name.
    Falls back to the original name if not in map.
    """
    return DISTRICT_MAP.get(db_name, db_name)
 
 
if __name__ == "__main__":
    print(f"Total mappings defined: {len(DISTRICT_MAP)}")
    print("\nSample mappings:")
    samples = list(DISTRICT_MAP.items())[:15]
    for db, shp in samples:
        print(f"  '{db}'  →  '{shp}'")