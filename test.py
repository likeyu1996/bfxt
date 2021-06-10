FCFF=887.89+360.08*0.7-(6430.01-6445.18)
'''
print(FCFF)
print(FCFF*1.12)
print(FCFF*1.12*1.12595)
print(FCFF*1.12*1.12595*1.00522)
#print(FCFF*1.12*1.12595*1.00522*0.72035)
print(FCFF*1.12*1.12595*1.00522*1.03)
'''
FCFF2020 = FCFF*1.12
FCFF2021 = FCFF2020*1.12595
FCFF2022 = FCFF2021*1.00522
FCFF2023 = FCFF2022*1.03
g=0.03
beta = 1.03
B = 3507.08+30.53+463.5
S = 3633.07
rf = 0.0322
rb = 390.02/B
print(B)
print(rb)
rm = (5629.06/3283.49)**(1/10)-1
print(rm)
# 国证A指
t = 203.29/843.61
print(t)
rs = rf + beta*(rm-rf)
print(rs)
wacc = S/(B+S)*rs + B/(B+S)*rb*(1-t)
print(wacc)
v2020=FCFF2021/(1+wacc)**1+ FCFF2022/(1+wacc)**2+ FCFF2023/(1+wacc)**3+ FCFF2023*(1+g)/(wacc-g)/(1+wacc)**3
print(v2020)
V2021=(FCFF2022/(1+wacc)**1+ FCFF2023/(1+wacc)**2+ FCFF2023*(1+g)/(wacc-g)/(1+wacc)**2)/(1+wacc)**1
print(V2021)
#sss = v2020+2466.92+142.09-10270.2-188.01
sss = v2020+2466.92+142.09-B-188.01
print(sss)
