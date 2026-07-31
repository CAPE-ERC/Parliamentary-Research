# Topic Classifier (Stage 2) - Validation Report

- Base model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- Training examples: 49248 (validation: 5472)
- Classes: 51 (50 Policy + non_policy catch-all)
- non_policy downsampled to: 5000

```
                                                      precision    recall  f1-score   support

                Air Mauritius, airports and aviation       0.81      0.72      0.76        81
                Arts, culture, artists and copyright       0.86      0.79      0.82        38
          Banking, financial sector and public funds       0.78      0.64      0.70       102
             Beaches, coastal management and erosion       0.82      0.61      0.70        46
    Budget and economic-policy debate (French/mixed)       0.79      0.80      0.79       260
           Bus services, routes and public transport       0.84      0.82      0.83        65
          CCTV, surveillance and security technology       0.75      0.85      0.80        48
            CSR, NGOs and social-development funding       0.71      0.72      0.72        69
                 Cancer statistics and annual trends       0.76      0.88      0.81        42
       Child protection, family welfare and shelters       0.79      0.78      0.79       111
Cooperatives and SME discourse with member exchanges       0.77      0.59      0.67        17
                Courts, prosecutions and legal cases       0.80      0.82      0.81       117
        Disability services, training and employment       0.76      0.66      0.71        44
                  Domestic and gender-based violence       0.88      0.73      0.80        30
             Drug control, trafficking and treatment       0.82      0.84      0.83       162
  Electric and motor vehicles with financial figures       0.67      0.76      0.72        38
             Electricity, energy and renewable power       0.84      0.88      0.86       139
                  Employment, workers and jobseekers       0.88      0.88      0.88        33
                            Fire and rescue services       0.87      0.85      0.86        40
             Fisheries, marine resources and vessels       0.79      0.90      0.84       126
 Foreign affairs, Chagos and international relations       0.70      0.67      0.68       203
       Gambling, horse racing and betting regulation       0.75      0.62      0.68        39
                    Hajj pilgrimage and Saudi Arabia       0.83      0.86      0.84        44
Healthcare services, hospitals and medical workforce       0.82      0.88      0.85       298
   Livestock, veterinary services and animal welfare       0.77      0.82      0.80        62
         MBC broadcasting, news and media governance       0.78      0.82      0.80        22
              Metro Express and light-rail transport       0.80      0.75      0.77        64
         Motor vehicles, imports and NLTA regulation       0.86      0.75      0.80        32
   Pensions, retirement and social-security benefits       0.78      0.84      0.81        76
                Petroleum prices, duties and imports       0.68      0.79      0.73        73
            Police investigations and criminal cases       0.80      0.85      0.82       190
     Police, prisons and home-affairs administration       0.76      0.60      0.67       124
             Port Louis local affairs and governance       0.65      0.68      0.67        92
                     Primary and secondary education       0.85      0.89      0.87       205
       Public buildings, construction and renovation       0.73      0.76      0.75       112
                   Public finance, banking and funds       0.72      0.84      0.78       151
              Road construction, traffic and bridges       0.76      0.82      0.79       217
          Road safety, driving and traffic accidents       0.77      0.66      0.71        65
       Rodrigues regional governance and development       0.82      0.88      0.85        73
         SMEs, entrepreneurship and business support       0.80      0.74      0.77        38
               Social housing and NHDC housing units       0.77      0.73      0.75        93
       Solid waste, landfill and plastics management       0.67      0.63      0.65        35
             Sport, football and athlete development       0.90      0.88      0.89       182
             Sports grounds, stadiums and facilities       0.72      0.80      0.76        49
          State land, leases and land administration       0.70      0.67      0.68       148
              Sugar, agriculture and food production       0.79      0.81      0.80       107
          Tertiary education and university training       0.74      0.67      0.70        72
                 Tourism, hotels and visitor economy       0.79      0.72      0.76        69
             Vaccination, COVID-19 and public health       0.81      0.87      0.84       146
  Water supply, drainage and sewerage infrastructure       0.81      0.83      0.82       283
                                          non_policy       0.84      0.77      0.80       500

                                            accuracy                           0.79      5472
                                           macro avg       0.78      0.77      0.78      5472
                                        weighted avg       0.79      0.79      0.79      5472

```
