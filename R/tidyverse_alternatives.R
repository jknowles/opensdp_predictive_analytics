# Tidyverse erratta

# Tidyverse frequency and prop tables
# TODO - how do you want to do these crosstabs in dplyr instead of in a table?
# sea_data %>% group_by(coop_name_g7) %>% 
#   count(male, name = "male_count") %>% 
#   mutate(male = ifelse(male == 1, "m", "f")) %>%
#   spread(male, male_count)
# 
# sea_data %>% group_by(coop_name_g7) %>%
#   count(frpl_7, name = "frpl_count") %>%
#   mutate(frpl_7 = paste0("frpl_", frpl_7)) %>%
#   spread(frpl_7, frpl_count)
# 
# # http://analyticswithr.com/contingencytables.html
# for(var in c("male", "race_ethnicity", "frpl_7", "iep_7", "ell_7",
#            "gifted_7")){
#   print(var)
#   # Print the table
#   sea_data %>%
#     group_by(coop_name_g7, get(var)) %>%
#     summarize(n=n()) %>% 
#     mutate(per = round(n / sum(n), 2)) %>%
#     kable() %>% # Format
#     print # print
# }

# sea_data %>% group_by(coop_name_g7) %>% 
#   count(any_grad) 
