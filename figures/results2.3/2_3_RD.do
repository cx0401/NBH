clear
ssc install rdrobust

use "/Users/admin/Desktop/demand_scores_agg.dta"

generate period_index = 0
replace period_index = 1 if (year == 2022 & week >= 35)|(year == 2023 & week <= 28)
drop if year == 2023 & week >= 29
drop if year == 2022 & week >= 29 & week <= 34


generate week_1 = 0
replace week_1 = week - 50 if year == 2021
replace week_1 = week + 2 if year == 2022

generate week_2 = 0
replace week_2 = week - 50 if year == 2022
replace week_2 = week + 2 if year == 2023

set scheme s1color
grstyle init
grstyle set symbol Oh
grstyle set legend 2, inside


rdplot s2_avg week_2 if period_index == 1, c(0) nbins(15 15) p(2) h(15 15) graph_options(ytitle(Demand shift, size(vlarge)) xtitle(Weeks, size(large)) legend(size(medium) order(2) label(2 "Polynomial fit of order 2")) )
graph export "labor_market_effects_1.png", as(png)
rdplot s2_avg week_1 if period_index == 0, c(0) nbins(15 15) p(2) h(15 15) graph_options(ytitle(Demand shift, size(vlarge)) xtitle(Weeks, size(large)) legend(size(medium) order(2) label(2 "Polynomial fit of order 2")) )
graph export "labor_market_effects_2.png", as(png)
rdplot s1_avg week_2 if period_index == 1, c(0) nbins(15 15) p(2) h(15 15) graph_options(ytitle(Demand shift, size(vlarge)) xtitle(Weeks, size(large)) legend(size(medium) order(2) label(2 "Polynomial fit of order 2")) )
graph export "labor_market_effects_3.png", as(png)
rdplot s1_avg week_1 if period_index == 0, c(0) nbins(15 15) p(2) h(15 15) graph_options(ytitle(Demand shift, size(vlarge)) xtitle(Weeks, size(large)) legend(size(medium) order(2) label(2 "Polynomial fit of order 2")) )
graph export "labor_market_effects_4.png", as(png)




