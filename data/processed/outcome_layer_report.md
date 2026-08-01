# Outcome Layer - Time-Series / Granger-Style Report (H3)

- Topics modeled: 50
- Sittings (chronological): 402
- Topics with enough conflict variation to test: 38 of 50 (need >= 10 sittings with conflict_rate > 0)

## Headline: lag-1 Granger results (does conflict_t predict attention_{t+1}?)

**3 of 38 topics significant at raw p<0.05** (1 survive Bonferroni correction, alpha=0.00132).

| Topic | N sittings | N conflict sittings | p-value (lag 1) |
|---|---|---|---|
| Vaccination, COVID-19 and public health | 402 | 38 | 0.0007 |
| Pensions, retirement and social-security benefits | 402 | 22 | 0.0100 |
| Public buildings, construction and renovation | 402 | 16 | 0.0294 |
| Banking, financial sector and public funds | 402 | 33 | 0.0704 |
| Road construction, traffic and bridges | 402 | 37 | 0.0729 |
| Metro Express and light-rail transport | 402 | 23 | 0.0801 |
| Primary and secondary education | 402 | 38 | 0.1293 |
| Sport, football and athlete development | 402 | 35 | 0.1526 |
| Police investigations and criminal cases | 402 | 61 | 0.1772 |
| Drug control, trafficking and treatment | 402 | 37 | 0.2264 |
| Tertiary education and university training | 402 | 17 | 0.2642 |
| Electricity, energy and renewable power | 402 | 30 | 0.3145 |
| Police, prisons and home-affairs administration | 402 | 29 | 0.3441 |
| Rodrigues regional governance and development | 402 | 25 | 0.3817 |
| Sugar, agriculture and food production | 402 | 30 | 0.4127 |
| Port Louis local affairs and governance | 402 | 62 | 0.4409 |
| Healthcare services, hospitals and medical workforce | 402 | 68 | 0.4707 |
| Electric and motor vehicles with financial figures | 402 | 23 | 0.5380 |
| Tourism, hotels and visitor economy | 402 | 14 | 0.5574 |
| Employment, workers and jobseekers | 402 | 11 | 0.5850 |
| Courts, prosecutions and legal cases | 402 | 72 | 0.6253 |
| Road safety, driving and traffic accidents | 402 | 26 | 0.6254 |
| Air Mauritius, airports and aviation | 402 | 24 | 0.6328 |
| CSR, NGOs and social-development funding | 402 | 16 | 0.7078 |
| Foreign affairs, Chagos and international relations | 402 | 121 | 0.7086 |
| Cancer statistics and annual trends | 402 | 17 | 0.7201 |
| Fisheries, marine resources and vessels | 402 | 31 | 0.7235 |
| CCTV, surveillance and security technology | 402 | 14 | 0.7360 |
| State land, leases and land administration | 402 | 38 | 0.7847 |
| Social housing and NHDC housing units | 402 | 26 | 0.8095 |
| Public finance, banking and funds | 402 | 80 | 0.8386 |
| Budget and economic-policy debate (French/mixed) | 402 | 154 | 0.8651 |
| Hajj pilgrimage and Saudi Arabia | 402 | 18 | 0.8655 |
| Bus services, routes and public transport | 402 | 15 | 0.9080 |
| Water supply, drainage and sewerage infrastructure | 402 | 54 | 0.9117 |
| Child protection, family welfare and shelters | 402 | 31 | 0.9446 |
| Disability services, training and employment | 402 | 12 | 0.9560 |
| Petroleum prices, duties and imports | 402 | 26 | 0.9756 |

## Lag 2-3 results (robustness)

| Topic | Lag | p-value |
|---|---|---|
| Air Mauritius, airports and aviation | 2.0 | 0.8817 |
| Air Mauritius, airports and aviation | 3.0 | 0.9741 |
| Banking, financial sector and public funds | 2.0 | 0.1777 |
| Banking, financial sector and public funds | 3.0 | 0.2232 |
| Budget and economic-policy debate (French/mixed) | 2.0 | 0.4840 |
| Budget and economic-policy debate (French/mixed) | 3.0 | 0.3931 |
| Bus services, routes and public transport | 2.0 | 0.9923 |
| Bus services, routes and public transport | 3.0 | 0.9971 |
| CCTV, surveillance and security technology | 2.0 | 0.6384 |
| CCTV, surveillance and security technology | 3.0 | 0.7943 |
| CSR, NGOs and social-development funding | 2.0 | 0.8956 |
| CSR, NGOs and social-development funding | 3.0 | 0.5844 |
| Cancer statistics and annual trends | 2.0 | 0.8949 |
| Cancer statistics and annual trends | 3.0 | 0.8952 |
| Child protection, family welfare and shelters | 2.0 | 0.9900 |
| Child protection, family welfare and shelters | 3.0 | 0.9291 |
| Courts, prosecutions and legal cases | 2.0 | 0.5675 |
| Courts, prosecutions and legal cases | 3.0 | 0.2168 |
| Disability services, training and employment | 2.0 | 0.8569 |
| Disability services, training and employment | 3.0 | 0.9540 |
| Drug control, trafficking and treatment | 2.0 | 0.2628 |
| Drug control, trafficking and treatment | 3.0 | 0.4843 |
| Electric and motor vehicles with financial figures | 2.0 | 0.4691 |
| Electric and motor vehicles with financial figures | 3.0 | 0.6625 |
| Electricity, energy and renewable power | 2.0 | 0.5888 |
| Electricity, energy and renewable power | 3.0 | 0.7899 |
| Employment, workers and jobseekers | 2.0 | 0.8626 |
| Employment, workers and jobseekers | 3.0 | 0.9609 |
| Fisheries, marine resources and vessels | 2.0 | 0.9259 |
| Fisheries, marine resources and vessels | 3.0 | 0.9418 |
| Foreign affairs, Chagos and international relations | 2.0 | 0.8745 |
| Foreign affairs, Chagos and international relations | 3.0 | 0.2446 |
| Hajj pilgrimage and Saudi Arabia | 2.0 | 0.9831 |
| Hajj pilgrimage and Saudi Arabia | 3.0 | 0.8142 |
| Healthcare services, hospitals and medical workforce | 2.0 | 0.6747 |
| Healthcare services, hospitals and medical workforce | 3.0 | 0.7934 |
| Metro Express and light-rail transport | 2.0 | 0.1763 |
| Metro Express and light-rail transport | 3.0 | 0.2797 |
| Pensions, retirement and social-security benefits | 2.0 | 0.0282 |
| Pensions, retirement and social-security benefits | 3.0 | 0.0657 |
| Petroleum prices, duties and imports | 2.0 | 0.9402 |
| Petroleum prices, duties and imports | 3.0 | 0.3475 |
| Police investigations and criminal cases | 2.0 | 0.3065 |
| Police investigations and criminal cases | 3.0 | 0.4812 |
| Police, prisons and home-affairs administration | 2.0 | 0.6049 |
| Police, prisons and home-affairs administration | 3.0 | 0.6515 |
| Port Louis local affairs and governance | 2.0 | 0.7832 |
| Port Louis local affairs and governance | 3.0 | 0.1572 |
| Primary and secondary education | 2.0 | 0.2835 |
| Primary and secondary education | 3.0 | 0.3487 |
| Public buildings, construction and renovation | 2.0 | 0.0095 |
| Public buildings, construction and renovation | 3.0 | 0.0001 |
| Public finance, banking and funds | 2.0 | 0.6427 |
| Public finance, banking and funds | 3.0 | 0.6155 |
| Road construction, traffic and bridges | 2.0 | 0.1876 |
| Road construction, traffic and bridges | 3.0 | 0.3533 |
| Road safety, driving and traffic accidents | 2.0 | 0.5986 |
| Road safety, driving and traffic accidents | 3.0 | 0.7366 |
| Rodrigues regional governance and development | 2.0 | 0.7062 |
| Rodrigues regional governance and development | 3.0 | 0.8115 |
| Social housing and NHDC housing units | 2.0 | 0.9416 |
| Social housing and NHDC housing units | 3.0 | 0.9770 |
| Sport, football and athlete development | 2.0 | 0.3268 |
| Sport, football and athlete development | 3.0 | 0.4690 |
| State land, leases and land administration | 2.0 | 0.1413 |
| State land, leases and land administration | 3.0 | 0.2164 |
| Sugar, agriculture and food production | 2.0 | 0.6464 |
| Sugar, agriculture and food production | 3.0 | 0.8077 |
| Tertiary education and university training | 2.0 | 0.3420 |
| Tertiary education and university training | 3.0 | 0.5267 |
| Tourism, hotels and visitor economy | 2.0 | 0.8440 |
| Tourism, hotels and visitor economy | 3.0 | 0.9362 |
| Vaccination, COVID-19 and public health | 2.0 | 0.0120 |
| Vaccination, COVID-19 and public health | 3.0 | 0.0233 |
| Water supply, drainage and sewerage infrastructure | 2.0 | 0.8735 |
| Water supply, drainage and sewerage infrastructure | 3.0 | 0.9407 |

## Pooled panel model (robustness cross-check)

`attention_share_(t+1) ~ attention_share_t + conflict_rate_t`, random intercept by topic, n=20050, converged=True

```
              Mixed Linear Model Regression Results
==================================================================
Model:            MixedLM Dependent Variable: attention_share_next
No. Observations: 20050   Method:             ML                  
No. Groups:       50      Scale:              0.0002              
Min. group size:  401     Log-Likelihood:     58979.3226          
Max. group size:  401     Converged:          Yes                 
Mean group size:  401.0                                           
-------------------------------------------------------------------
                    Coef.   Std.Err.    z     P>|z|  [0.025  0.975]
-------------------------------------------------------------------
Intercept            0.006     0.001   8.777  0.000   0.004   0.007
attention_share      0.177     0.007  25.344  0.000   0.163   0.191
conflict_rate       -0.000     0.001  -0.130  0.897  -0.002   0.001
Group Var            0.000     0.000                               
==================================================================

```

## Caveats

- **Stationarity not tested.** Granger causality assumes stationary series; a 2015-2025 span plausibly has trends/structural breaks (e.g. COVID-era health-topic spikes). Full per-topic ADF unit-root testing wasn't run given scope (50 series) - treat results as suggestive, not confirmatory, without that check.
- 12 topics had fewer than 10 sittings with nonzero conflict_rate and were skipped - too little variation in the predictor to test meaningfully, not evidence of no effect for those topics specifically.
- With 47-50 near-independent per-topic tests, some raw p<0.05 results are expected by chance alone (~2-2.5 at a 5% rate) - the Bonferroni-adjusted count is the more trustworthy summary statistic.
