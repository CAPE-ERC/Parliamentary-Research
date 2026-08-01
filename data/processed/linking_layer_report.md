# Linking Layer - Regression Report

- Panel rows used: 4735 (of 4735 total; 0 topics dropped for having fewer than 10 panel cells)
- Distinct topics modeled: 48
- Distinct chairs (random effect groups): 4
- Converged: True

## Headline: H1 does not find broad evidence of asymmetric enforcement

Only 1 of 47 topic x opposition interaction terms is significant at p<0.05
("Vaccination, COVID-19 and public health," coefficient -0.143, p=0.044) -
and that is at essentially chance level: with 47 independent tests at
alpha=0.05, ~2.35 false positives are expected under the null even if there
is no real effect anywhere. Nothing survives a Bonferroni correction
(adjusted alpha = 0.05/47 = 0.0011; the smallest p-value found is 0.025).
The one nominally-significant term also points the *opposite* direction
from H1's prediction (opposition utterances on that topic show a LOWER
intervention rate, not higher), and COVID/vaccination is not one of the
"corruption, procurement, institutional scandal" topics H1 was framed
around in the first place.

**This is a real null result, not an underpowered one.** Before the
speaker-registry work, this test would have run on 145 opposition-tagged
utterances (all from one office) - not powered to find anything either
way. With the registry resolving backbench MPs, the panel now has 13,032
opposition-attributed utterances across 4 chairs and 48 topics (see
Section 5.3 methodology below), and still finds no broad pattern of
topic-conditional asymmetric intervention. That is itself a meaningful,
reportable finding for the study, not a methodology failure.

## H2: PNQ deflection - inconclusive due to event sparsity, not a null result

pnq_transfer is a genuinely rare event (226 of 211,567 utterances corpus-
wide). Broken down by topic, most cells have 0-2 transfer events (see table
below) - there isn't enough signal per topic to say anything about whether
sensitive topics see disproportionate deflection. Unlike H1, this is not a
confident null - it's "not enough events to tell," and would need either a
coarser grouping (e.g. Policy vs. Governance/Procedure rather than 50
individual topics) or a larger corpus to test properly.

## H1: topic x party interaction terms (asymmetric enforcement)

| Term | Coefficient | p-value |
|---|---|---|
| C(topic)[T.Arts, culture, artists and copyright]:C(party)[T.opposition] | -0.1130 | 0.2230 |
| C(topic)[T.Banking, financial sector and public funds]:C(party)[T.opposition] | -0.1077 | 0.1317 |
| C(topic)[T.Beaches, coastal management and erosion]:C(party)[T.opposition] | -0.0602 | 0.5525 |
| C(topic)[T.Budget and economic-policy debate (French/mixed)]:C(party)[T.opposition] | -0.0310 | 0.6229 |
| C(topic)[T.Bus services, routes and public transport]:C(party)[T.opposition] | -0.0730 | 0.3970 |
| C(topic)[T.CCTV, surveillance and security technology]:C(party)[T.opposition] | 0.0126 | 0.8912 |
| C(topic)[T.CSR, NGOs and social-development funding]:C(party)[T.opposition] | -0.0710 | 0.3661 |
| C(topic)[T.Cancer statistics and annual trends]:C(party)[T.opposition] | 0.0778 | 0.5422 |
| C(topic)[T.Child protection, family welfare and shelters]:C(party)[T.opposition] | 0.0230 | 0.7534 |
| C(topic)[T.Courts, prosecutions and legal cases]:C(party)[T.opposition] | -0.0219 | 0.7461 |
| C(topic)[T.Disability services, training and employment]:C(party)[T.opposition] | -0.0589 | 0.5157 |
| C(topic)[T.Domestic and gender-based violence]:C(party)[T.opposition] | -0.1119 | 0.4789 |
| C(topic)[T.Drug control, trafficking and treatment]:C(party)[T.opposition] | -0.0298 | 0.6621 |
| C(topic)[T.Electric and motor vehicles with financial figures]:C(party)[T.opposition] | -0.0303 | 0.7849 |
| C(topic)[T.Electricity, energy and renewable power]:C(party)[T.opposition] | -0.0655 | 0.3190 |
| C(topic)[T.Fire and rescue services]:C(party)[T.opposition] | -0.0584 | 0.5783 |
| C(topic)[T.Fisheries, marine resources and vessels]:C(party)[T.opposition] | -0.0219 | 0.7582 |
| C(topic)[T.Foreign affairs, Chagos and international relations]:C(party)[T.opposition] | -0.0697 | 0.2707 |
| C(topic)[T.Gambling, horse racing and betting regulation]:C(party)[T.opposition] | -0.0815 | 0.4482 |
| C(topic)[T.Hajj pilgrimage and Saudi Arabia]:C(party)[T.opposition] | -0.0109 | 0.8957 |
| C(topic)[T.Healthcare services, hospitals and medical workforce]:C(party)[T.opposition] | -0.0496 | 0.4250 |
| C(topic)[T.Livestock, veterinary services and animal welfare]:C(party)[T.opposition] | -0.0317 | 0.6980 |
| C(topic)[T.MBC broadcasting, news and media governance]:C(party)[T.opposition] | -0.0141 | 0.9125 |
| C(topic)[T.Metro Express and light-rail transport]:C(party)[T.opposition] | -0.1799 | 0.0519 |
| C(topic)[T.Motor vehicles, imports and NLTA regulation]:C(party)[T.opposition] | -0.1100 | 0.2464 |
| C(topic)[T.Pensions, retirement and social-security benefits]:C(party)[T.opposition] | 0.0922 | 0.2560 |
| C(topic)[T.Petroleum prices, duties and imports]:C(party)[T.opposition] | -0.0173 | 0.8314 |
| C(topic)[T.Police investigations and criminal cases]:C(party)[T.opposition] | -0.0114 | 0.8564 |
| C(topic)[T.Police, prisons and home-affairs administration]:C(party)[T.opposition] | -0.0650 | 0.4499 |
| C(topic)[T.Port Louis local affairs and governance]:C(party)[T.opposition] | -0.0928 | 0.5413 |
| C(topic)[T.Primary and secondary education]:C(party)[T.opposition] | -0.0444 | 0.4959 |
| C(topic)[T.Public buildings, construction and renovation]:C(party)[T.opposition] | -0.0928 | 0.1999 |
| C(topic)[T.Public finance, banking and funds]:C(party)[T.opposition] | -0.1045 | 0.0932 |
| C(topic)[T.Road construction, traffic and bridges]:C(party)[T.opposition] | 0.0019 | 0.9762 |
| C(topic)[T.Road safety, driving and traffic accidents]:C(party)[T.opposition] | -0.1106 | 0.2184 |
| C(topic)[T.Rodrigues regional governance and development]:C(party)[T.opposition] | -0.0196 | 0.8208 |
| C(topic)[T.SMEs, entrepreneurship and business support]:C(party)[T.opposition] | -0.0042 | 0.9649 |
| C(topic)[T.Social housing and NHDC housing units]:C(party)[T.opposition] | -0.0319 | 0.6740 |
| C(topic)[T.Solid waste, landfill and plastics management]:C(party)[T.opposition] | -0.1548 | 0.1452 |
| C(topic)[T.Sport, football and athlete development]:C(party)[T.opposition] | -0.0375 | 0.5671 |
| C(topic)[T.Sports grounds, stadiums and facilities]:C(party)[T.opposition] | -0.0846 | 0.4272 |
| C(topic)[T.State land, leases and land administration]:C(party)[T.opposition] | -0.1046 | 0.1140 |
| C(topic)[T.Sugar, agriculture and food production]:C(party)[T.opposition] | -0.0656 | 0.3649 |
| C(topic)[T.Tertiary education and university training]:C(party)[T.opposition] | -0.0383 | 0.6418 |
| C(topic)[T.Tourism, hotels and visitor economy]:C(party)[T.opposition] | -0.1485 | 0.1072 |
| C(topic)[T.Vaccination, COVID-19 and public health]:C(party)[T.opposition] | -0.1435 | 0.0439 |
| C(topic)[T.Water supply, drainage and sewerage infrastructure]:C(party)[T.opposition] | -0.0618 | 0.3283 |

1 of 47 topic x opposition interaction terms significant at p<0.05.

## H2: PNQ deflection rate by topic (secondary analysis)

| Topic | N | N transferred | Transfer rate |
|---|---|---|---|
| Road safety, driving and traffic accidents | 682 | 2 | 0.0029 |
| SMEs, entrepreneurship and business support | 486 | 1 | 0.0021 |
| Sports grounds, stadiums and facilities | 553 | 1 | 0.0018 |
| Tertiary education and university training | 841 | 1 | 0.0012 |
| CSR, NGOs and social-development funding | 941 | 1 | 0.0011 |
| Port Louis local affairs and governance | 1054 | 1 | 0.0009 |
| Drug control, trafficking and treatment | 1704 | 1 | 0.0006 |
| Foreign affairs, Chagos and international relations | 2729 | 1 | 0.0004 |
| Arts, culture, artists and copyright | 466 | 0 | 0.0000 |
| Air Mauritius, airports and aviation | 949 | 0 | 0.0000 |
| Child protection, family welfare and shelters | 1289 | 0 | 0.0000 |
| Cooperatives and SME discourse with member exchanges | 170 | 0 | 0.0000 |
| Budget and economic-policy debate (French/mixed) | 3147 | 0 | 0.0000 |
| Banking, financial sector and public funds | 1259 | 0 | 0.0000 |
| Bus services, routes and public transport | 778 | 0 | 0.0000 |


Full model summary:

```
                                              Mixed Linear Model Regression Results
==================================================================================================================================
Model:                                   MixedLM                       Dependent Variable:                       intervention_rate
No. Observations:                        4735                          Method:                                   ML               
No. Groups:                              4                             Scale:                                    0.0536           
Min. group size:                         31                            Log-Likelihood:                           204.6788         
Max. group size:                         2507                          Converged:                                Yes              
Mean group size:                         1183.8                                                                                   
----------------------------------------------------------------------------------------------------------------------------------
                                                                                        Coef.  Std.Err.   z    P>|z| [0.025 0.975]
----------------------------------------------------------------------------------------------------------------------------------
Intercept                                                                                0.190    0.041  4.679 0.000  0.110  0.269
C(topic)[T.Arts, culture, artists and copyright]                                        -0.134    0.059 -2.248 0.025 -0.250 -0.017
C(topic)[T.Banking, financial sector and public funds]                                   0.018    0.044  0.400 0.689 -0.069  0.104
C(topic)[T.Beaches, coastal management and erosion]                                     -0.087    0.065 -1.331 0.183 -0.214  0.041
C(topic)[T.Budget and economic-policy debate (French/mixed)]                             0.236    0.037  6.357 0.000  0.163  0.308
C(topic)[T.Bus services, routes and public transport]                                   -0.048    0.049 -0.987 0.324 -0.143  0.047
C(topic)[T.CCTV, surveillance and security technology]                                  -0.101    0.054 -1.859 0.063 -0.207  0.005
C(topic)[T.CSR, NGOs and social-development funding]                                    -0.063    0.046 -1.356 0.175 -0.153  0.028
C(topic)[T.Cancer statistics and annual trends]                                         -0.118    0.060 -1.946 0.052 -0.236  0.001
C(topic)[T.Child protection, family welfare and shelters]                               -0.084    0.044 -1.927 0.054 -0.170  0.001
C(topic)[T.Courts, prosecutions and legal cases]                                         0.045    0.039  1.154 0.249 -0.032  0.122
C(topic)[T.Disability services, training and employment]                                -0.050    0.058 -0.861 0.389 -0.163  0.063
C(topic)[T.Domestic and gender-based violence]                                          -0.073    0.072 -1.008 0.313 -0.214  0.069
C(topic)[T.Drug control, trafficking and treatment]                                      0.018    0.041  0.429 0.668 -0.063  0.098
C(topic)[T.Electric and motor vehicles with financial figures]                           0.163    0.060  2.691 0.007  0.044  0.281
C(topic)[T.Electricity, energy and renewable power]                                     -0.033    0.041 -0.819 0.413 -0.113  0.046
C(topic)[T.Fire and rescue services]                                                    -0.080    0.061 -1.303 0.193 -0.200  0.040
C(topic)[T.Fisheries, marine resources and vessels]                                     -0.027    0.042 -0.642 0.521 -0.110  0.056
C(topic)[T.Foreign affairs, Chagos and international relations]                          0.052    0.037  1.414 0.157 -0.020  0.125
C(topic)[T.Gambling, horse racing and betting regulation]                                0.038    0.065  0.583 0.560 -0.090  0.166
C(topic)[T.Hajj pilgrimage and Saudi Arabia]                                             0.027    0.052  0.511 0.610 -0.076  0.129
C(topic)[T.Healthcare services, hospitals and medical workforce]                        -0.060    0.038 -1.576 0.115 -0.134  0.015
C(topic)[T.Livestock, veterinary services and animal welfare]                           -0.144    0.053 -2.745 0.006 -0.248 -0.041
C(topic)[T.MBC broadcasting, news and media governance]                                 -0.007    0.075 -0.090 0.929 -0.153  0.139
C(topic)[T.Metro Express and light-rail transport]                                       0.094    0.053  1.773 0.076 -0.010  0.198
C(topic)[T.Motor vehicles, imports and NLTA regulation]                                 -0.013    0.059 -0.213 0.831 -0.129  0.104
C(topic)[T.Pensions, retirement and social-security benefits]                           -0.050    0.047 -1.061 0.289 -0.143  0.043
C(topic)[T.Petroleum prices, duties and imports]                                         0.009    0.045  0.208 0.835 -0.079  0.097
C(topic)[T.Police investigations and criminal cases]                                     0.007    0.038  0.193 0.847 -0.067  0.082
C(topic)[T.Police, prisons and home-affairs administration]                             -0.037    0.060 -0.614 0.539 -0.155  0.081
C(topic)[T.Port Louis local affairs and governance]                                      0.077    0.088  0.871 0.384 -0.096  0.250
C(topic)[T.Primary and secondary education]                                             -0.141    0.040 -3.556 0.000 -0.219 -0.063
C(topic)[T.Public buildings, construction and renovation]                               -0.056    0.044 -1.268 0.205 -0.143  0.031
C(topic)[T.Public finance, banking and funds]                                            0.039    0.039  1.007 0.314 -0.037  0.115
C(topic)[T.Road construction, traffic and bridges]                                      -0.063    0.038 -1.650 0.099 -0.138  0.012
C(topic)[T.Road safety, driving and traffic accidents]                                  -0.058    0.051 -1.146 0.252 -0.158  0.041
C(topic)[T.Rodrigues regional governance and development]                                0.030    0.054  0.561 0.575 -0.075  0.136
C(topic)[T.SMEs, entrepreneurship and business support]                                 -0.080    0.055 -1.445 0.149 -0.189  0.029
C(topic)[T.Social housing and NHDC housing units]                                        0.007    0.046  0.160 0.873 -0.083  0.097
C(topic)[T.Solid waste, landfill and plastics management]                               -0.060    0.059 -1.033 0.301 -0.175  0.054
C(topic)[T.Sport, football and athlete development]                                     -0.084    0.040 -2.086 0.037 -0.162 -0.005
C(topic)[T.Sports grounds, stadiums and facilities]                                     -0.116    0.064 -1.816 0.069 -0.241  0.009
C(topic)[T.State land, leases and land administration]                                  -0.043    0.040 -1.073 0.283 -0.121  0.035
C(topic)[T.Sugar, agriculture and food production]                                      -0.052    0.043 -1.221 0.222 -0.136  0.032
C(topic)[T.Tertiary education and university training]                                  -0.130    0.045 -2.879 0.004 -0.219 -0.042
C(topic)[T.Tourism, hotels and visitor economy]                                         -0.065    0.050 -1.317 0.188 -0.163  0.032
C(topic)[T.Vaccination, COVID-19 and public health]                                      0.048    0.042  1.152 0.249 -0.034  0.131
C(topic)[T.Water supply, drainage and sewerage infrastructure]                          -0.043    0.038 -1.139 0.255 -0.117  0.031
C(party)[T.opposition]                                                                   0.050    0.055  0.916 0.360 -0.057  0.158
C(topic)[T.Arts, culture, artists and copyright]:C(party)[T.opposition]                 -0.113    0.093 -1.219 0.223 -0.295  0.069
C(topic)[T.Banking, financial sector and public funds]:C(party)[T.opposition]           -0.108    0.071 -1.507 0.132 -0.248  0.032
C(topic)[T.Beaches, coastal management and erosion]:C(party)[T.opposition]              -0.060    0.101 -0.594 0.553 -0.259  0.138
C(topic)[T.Budget and economic-policy debate (French/mixed)]:C(party)[T.opposition]     -0.031    0.063 -0.492 0.623 -0.155  0.093
C(topic)[T.Bus services, routes and public transport]:C(party)[T.opposition]            -0.073    0.086 -0.847 0.397 -0.242  0.096
C(topic)[T.CCTV, surveillance and security technology]:C(party)[T.opposition]            0.013    0.092  0.137 0.891 -0.167  0.192
C(topic)[T.CSR, NGOs and social-development funding]:C(party)[T.opposition]             -0.071    0.079 -0.904 0.366 -0.225  0.083
C(topic)[T.Cancer statistics and annual trends]:C(party)[T.opposition]                   0.078    0.128  0.609 0.542 -0.172  0.328
C(topic)[T.Child protection, family welfare and shelters]:C(party)[T.opposition]         0.023    0.073  0.314 0.753 -0.120  0.166
C(topic)[T.Courts, prosecutions and legal cases]:C(party)[T.opposition]                 -0.022    0.068 -0.324 0.746 -0.154  0.111
C(topic)[T.Disability services, training and employment]:C(party)[T.opposition]         -0.059    0.091 -0.650 0.516 -0.237  0.119
C(topic)[T.Domestic and gender-based violence]:C(party)[T.opposition]                   -0.112    0.158 -0.708 0.479 -0.422  0.198
C(topic)[T.Drug control, trafficking and treatment]:C(party)[T.opposition]              -0.030    0.068 -0.437 0.662 -0.164  0.104
C(topic)[T.Electric and motor vehicles with financial figures]:C(party)[T.opposition]   -0.030    0.111 -0.273 0.785 -0.248  0.187
C(topic)[T.Electricity, energy and renewable power]:C(party)[T.opposition]              -0.066    0.066 -0.996 0.319 -0.194  0.063
C(topic)[T.Fire and rescue services]:C(party)[T.opposition]                             -0.058    0.105 -0.556 0.578 -0.264  0.148
C(topic)[T.Fisheries, marine resources and vessels]:C(party)[T.opposition]              -0.022    0.071 -0.308 0.758 -0.162  0.118
C(topic)[T.Foreign affairs, Chagos and international relations]:C(party)[T.opposition]  -0.070    0.063 -1.101 0.271 -0.194  0.054
C(topic)[T.Gambling, horse racing and betting regulation]:C(party)[T.opposition]        -0.081    0.107 -0.758 0.448 -0.292  0.129
C(topic)[T.Hajj pilgrimage and Saudi Arabia]:C(party)[T.opposition]                     -0.011    0.083 -0.131 0.896 -0.174  0.153
C(topic)[T.Healthcare services, hospitals and medical workforce]:C(party)[T.opposition] -0.050    0.062 -0.798 0.425 -0.172  0.072
C(topic)[T.Livestock, veterinary services and animal welfare]:C(party)[T.opposition]    -0.032    0.082 -0.388 0.698 -0.192  0.128
C(topic)[T.MBC broadcasting, news and media governance]:C(party)[T.opposition]          -0.014    0.128 -0.110 0.913 -0.265  0.237
C(topic)[T.Metro Express and light-rail transport]:C(party)[T.opposition]               -0.180    0.093 -1.944 0.052 -0.361  0.001
C(topic)[T.Motor vehicles, imports and NLTA regulation]:C(party)[T.opposition]          -0.110    0.095 -1.159 0.246 -0.296  0.076
C(topic)[T.Pensions, retirement and social-security benefits]:C(party)[T.opposition]     0.092    0.081  1.136 0.256 -0.067  0.251
C(topic)[T.Petroleum prices, duties and imports]:C(party)[T.opposition]                 -0.017    0.081 -0.213 0.831 -0.177  0.142
C(topic)[T.Police investigations and criminal cases]:C(party)[T.opposition]             -0.011    0.063 -0.181 0.856 -0.135  0.112
C(topic)[T.Police, prisons and home-affairs administration]:C(party)[T.opposition]      -0.065    0.086 -0.756 0.450 -0.234  0.104
C(topic)[T.Port Louis local affairs and governance]:C(party)[T.opposition]              -0.093    0.152 -0.611 0.541 -0.391  0.205
C(topic)[T.Primary and secondary education]:C(party)[T.opposition]                      -0.044    0.065 -0.681 0.496 -0.172  0.083
C(topic)[T.Public buildings, construction and renovation]:C(party)[T.opposition]        -0.093    0.072 -1.282 0.200 -0.235  0.049
C(topic)[T.Public finance, banking and funds]:C(party)[T.opposition]                    -0.104    0.062 -1.679 0.093 -0.226  0.017
C(topic)[T.Road construction, traffic and bridges]:C(party)[T.opposition]                0.002    0.065  0.030 0.976 -0.125  0.128
C(topic)[T.Road safety, driving and traffic accidents]:C(party)[T.opposition]           -0.111    0.090 -1.231 0.218 -0.287  0.066
C(topic)[T.Rodrigues regional governance and development]:C(party)[T.opposition]        -0.020    0.086 -0.226 0.821 -0.189  0.150
C(topic)[T.SMEs, entrepreneurship and business support]:C(party)[T.opposition]          -0.004    0.095 -0.044 0.965 -0.191  0.183
C(topic)[T.Social housing and NHDC housing units]:C(party)[T.opposition]                -0.032    0.076 -0.421 0.674 -0.180  0.117
C(topic)[T.Solid waste, landfill and plastics management]:C(party)[T.opposition]        -0.155    0.106 -1.457 0.145 -0.363  0.053
C(topic)[T.Sport, football and athlete development]:C(party)[T.opposition]              -0.037    0.065 -0.572 0.567 -0.166  0.091
C(topic)[T.Sports grounds, stadiums and facilities]:C(party)[T.opposition]              -0.085    0.107 -0.794 0.427 -0.293  0.124
C(topic)[T.State land, leases and land administration]:C(party)[T.opposition]           -0.105    0.066 -1.581 0.114 -0.234  0.025
C(topic)[T.Sugar, agriculture and food production]:C(party)[T.opposition]               -0.066    0.072 -0.906 0.365 -0.207  0.076
C(topic)[T.Tertiary education and university training]:C(party)[T.opposition]           -0.038    0.082 -0.465 0.642 -0.200  0.123
C(topic)[T.Tourism, hotels and visitor economy]:C(party)[T.opposition]                  -0.149    0.092 -1.611 0.107 -0.329  0.032
C(topic)[T.Vaccination, COVID-19 and public health]:C(party)[T.opposition]              -0.143    0.071 -2.015 0.044 -0.283 -0.004
C(topic)[T.Water supply, drainage and sewerage infrastructure]:C(party)[T.opposition]   -0.062    0.063 -0.978 0.328 -0.186  0.062
Group Var                                                                                0.002    0.006                           
==================================================================================================================================

```
