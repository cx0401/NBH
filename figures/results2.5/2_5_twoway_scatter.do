clear all
use "/Users/admin/Desktop/less than 10 years.dta"
append using "/Users/admin/Desktop/more than 10 years.dta"
rename avg_diff_salary Premium
rename rating Exposure
twoway (scatter Premium Exposure)|| lfit Premium Exposure ||, by(exp_group)
