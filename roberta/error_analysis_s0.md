### RoBERTa 151-class (cutoff 0.17)

- in-scope wrong intent: 131 | in-scope refused: 18 | OOS accepted: 363

- top confused pairs (gold -> pred): apr -> interest_rate (6); pin_change -> change_ai_name (5); redeem_rewards -> rewards_balance (5); calendar -> calendar_update (4); transactions -> spending_history (3); pay_bill -> bill_due (3); schedule_maintenance -> oil_change_when (3); reminder_update -> make_call (3); accept_reservations -> restaurant_reservation (3); card_declined -> damaged_card (3)
- OOS accepted as: directions (17); order (17); sync_device (12); nutrition_info (10); car_rental (9); smart_home (8); mpg (8); weather (8); jump_start (8); travel_suggestion (8)
- in-scope intents refused most: apr (3); replacement_card_duration (2); measurement_conversion (2); order (2); timer (1); insurance_change (1); where_are_you_from (1); travel_suggestion (1); carry_on (1); plug_type (1)

### RoBERTa 150-class + cutoff (cutoff 0.79)

- in-scope wrong intent: 89 | in-scope refused: 100 | OOS accepted: 334

- top confused pairs (gold -> pred): calendar -> calendar_update (4); redeem_rewards -> rewards_balance (4); pay_bill -> bill_due (3); schedule_maintenance -> oil_change_when (3); reminder_update -> make_call (3); restaurant_reservation -> accept_reservations (2); transactions -> spending_history (2); oil_change_when -> oil_change_how (2); change_volume -> cancel (2); how_busy -> restaurant_reservation (2)
- OOS accepted as: order (19); directions (15); travel_suggestion (11); weather (11); nutrition_info (11); who_made_you (10); how_old_are_you (10); who_do_you_work_for (10); text (9); sync_device (9)
- in-scope intents refused most: bill_balance (5); card_declined (5); replacement_card_duration (4); pin_change (4); order (4); pto_request (3); distance (3); transactions (3); yes (3); calendar (3)

