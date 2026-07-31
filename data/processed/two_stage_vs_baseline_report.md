# Two-Stage vs Single-Stage Classifier - Fair Comparison

Both models evaluated on the SAME held-out set, drawn from the full, naturally-distributed corpus (not the downsampled training pool), so this is apples-to-apples. Held-out set: 15992 examples, 68.9% non_policy (true corpus prevalence).

## Headline numbers

| Model | Accuracy | Macro F1 |
|---|---|---|
| Single-stage baseline (51-class) | 0.792 | 0.644 |
| Two-stage (gate + domain classifier) | 0.847 | 0.680 |

Training time: gate 192.3 min, domain classifier 145.9 min

## Two-stage pipeline - full classification report

```
                                                      precision    recall  f1-score   support

                Air Mauritius, airports and aviation       0.67      0.77      0.72        81
                Arts, culture, artists and copyright       0.55      0.68      0.61        38
          Banking, financial sector and public funds       0.48      0.62      0.54       102
             Beaches, coastal management and erosion       0.62      0.61      0.62        46
    Budget and economic-policy debate (French/mixed)       0.59      0.71      0.64       260
           Bus services, routes and public transport       0.69      0.80      0.74        65
          CCTV, surveillance and security technology       0.62      0.81      0.70        48
            CSR, NGOs and social-development funding       0.52      0.70      0.60        69
                 Cancer statistics and annual trends       0.60      0.57      0.59        42
       Child protection, family welfare and shelters       0.67      0.75      0.71       111
Cooperatives and SME discourse with member exchanges       0.85      0.65      0.73        17
                Courts, prosecutions and legal cases       0.52      0.71      0.60       117
        Disability services, training and employment       0.67      0.55      0.60        44
                  Domestic and gender-based violence       0.72      0.70      0.71        30
             Drug control, trafficking and treatment       0.78      0.83      0.80       162
  Electric and motor vehicles with financial figures       0.50      0.61      0.55        38
             Electricity, energy and renewable power       0.69      0.86      0.77       140
                  Employment, workers and jobseekers       0.69      0.75      0.72        32
                            Fire and rescue services       0.78      0.80      0.79        40
             Fisheries, marine resources and vessels       0.79      0.87      0.83       126
 Foreign affairs, Chagos and international relations       0.48      0.62      0.54       203
       Gambling, horse racing and betting regulation       0.55      0.59      0.57        39
                    Hajj pilgrimage and Saudi Arabia       0.58      0.86      0.70        44
Healthcare services, hospitals and medical workforce       0.74      0.85      0.79       298
   Livestock, veterinary services and animal welfare       0.70      0.84      0.76        62
         MBC broadcasting, news and media governance       0.52      0.77      0.62        22
              Metro Express and light-rail transport       0.64      0.77      0.70        64
         Motor vehicles, imports and NLTA regulation       0.69      0.75      0.72        32
   Pensions, retirement and social-security benefits       0.68      0.80      0.74        75
                Petroleum prices, duties and imports       0.48      0.67      0.56        73
            Police investigations and criminal cases       0.66      0.82      0.73       190
     Police, prisons and home-affairs administration       0.56      0.51      0.53       124
             Port Louis local affairs and governance       0.48      0.63      0.54        92
                     Primary and secondary education       0.73      0.89      0.80       206
       Public buildings, construction and renovation       0.53      0.64      0.58       112
                   Public finance, banking and funds       0.52      0.78      0.62       151
              Road construction, traffic and bridges       0.66      0.78      0.72       217
          Road safety, driving and traffic accidents       0.60      0.55      0.58        65
       Rodrigues regional governance and development       0.72      0.85      0.78        73
         SMEs, entrepreneurship and business support       0.61      0.74      0.67        38
               Social housing and NHDC housing units       0.67      0.74      0.70        93
       Solid waste, landfill and plastics management       0.52      0.66      0.58        35
             Sport, football and athlete development       0.75      0.88      0.81       182
             Sports grounds, stadiums and facilities       0.62      0.73      0.67        49
          State land, leases and land administration       0.58      0.64      0.60       148
              Sugar, agriculture and food production       0.66      0.82      0.73       107
          Tertiary education and university training       0.55      0.61      0.58        72
                 Tourism, hotels and visitor economy       0.65      0.74      0.69        69
             Vaccination, COVID-19 and public health       0.76      0.85      0.80       146
  Water supply, drainage and sewerage infrastructure       0.72      0.81      0.77       283
                                          non_policy       0.97      0.89      0.93     11020

                                            accuracy                           0.85     15992
                                           macro avg       0.64      0.73      0.68     15992
                                        weighted avg       0.87      0.85      0.85     15992

```

## Single-stage baseline - full classification report (same held-out set)

```
                                                      precision    recall  f1-score   support

                Air Mauritius, airports and aviation       0.62      0.73      0.67        81
                Arts, culture, artists and copyright       0.53      0.82      0.64        38
          Banking, financial sector and public funds       0.53      0.64      0.58       102
             Beaches, coastal management and erosion       0.62      0.61      0.62        46
    Budget and economic-policy debate (French/mixed)       0.43      0.80      0.56       260
           Bus services, routes and public transport       0.69      0.82      0.75        65
          CCTV, surveillance and security technology       0.47      0.85      0.61        48
            CSR, NGOs and social-development funding       0.46      0.72      0.56        69
                 Cancer statistics and annual trends       0.43      0.88      0.58        42
       Child protection, family welfare and shelters       0.60      0.78      0.68       111
Cooperatives and SME discourse with member exchanges       0.28      0.59      0.38        17
                Courts, prosecutions and legal cases       0.40      0.82      0.53       117
        Disability services, training and employment       0.48      0.66      0.56        44
                  Domestic and gender-based violence       0.58      0.73      0.65        30
             Drug control, trafficking and treatment       0.72      0.84      0.77       162
  Electric and motor vehicles with financial figures       0.39      0.76      0.52        38
             Electricity, energy and renewable power       0.60      0.89      0.72       140
                  Employment, workers and jobseekers       0.41      0.88      0.55        32
                            Fire and rescue services       0.72      0.85      0.78        40
             Fisheries, marine resources and vessels       0.69      0.90      0.78       126
 Foreign affairs, Chagos and international relations       0.38      0.67      0.48       203
       Gambling, horse racing and betting regulation       0.45      0.62      0.52        39
                    Hajj pilgrimage and Saudi Arabia       0.60      0.86      0.71        44
Healthcare services, hospitals and medical workforce       0.67      0.88      0.76       298
   Livestock, veterinary services and animal welfare       0.66      0.82      0.73        62
         MBC broadcasting, news and media governance       0.55      0.82      0.65        22
              Metro Express and light-rail transport       0.70      0.75      0.72        64
         Motor vehicles, imports and NLTA regulation       0.71      0.75      0.73        32
   Pensions, retirement and social-security benefits       0.49      0.84      0.62        75
                Petroleum prices, duties and imports       0.40      0.79      0.53        73
            Police investigations and criminal cases       0.56      0.85      0.67       190
     Police, prisons and home-affairs administration       0.44      0.60      0.51       124
             Port Louis local affairs and governance       0.42      0.68      0.52        92
                     Primary and secondary education       0.67      0.89      0.77       206
       Public buildings, construction and renovation       0.39      0.76      0.52       112
                   Public finance, banking and funds       0.41      0.84      0.55       151
              Road construction, traffic and bridges       0.61      0.82      0.70       217
          Road safety, driving and traffic accidents       0.53      0.66      0.59        65
       Rodrigues regional governance and development       0.70      0.88      0.78        73
         SMEs, entrepreneurship and business support       0.54      0.74      0.62        38
               Social housing and NHDC housing units       0.60      0.73      0.66        93
       Solid waste, landfill and plastics management       0.48      0.63      0.54        35
             Sport, football and athlete development       0.68      0.88      0.77       182
             Sports grounds, stadiums and facilities       0.67      0.80      0.73        49
          State land, leases and land administration       0.50      0.67      0.57       148
              Sugar, agriculture and food production       0.67      0.81      0.74       107
          Tertiary education and university training       0.48      0.67      0.55        72
                 Tourism, hotels and visitor economy       0.68      0.72      0.70        69
             Vaccination, COVID-19 and public health       0.71      0.87      0.78       146
  Water supply, drainage and sewerage infrastructure       0.65      0.83      0.73       283
                                          non_policy       0.99      0.79      0.88     11020

                                            accuracy                           0.79     15992
                                           macro avg       0.56      0.77      0.64     15992
                                        weighted avg       0.86      0.79      0.81     15992

```
