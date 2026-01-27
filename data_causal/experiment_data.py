

import pandas as pd
import numpy as np
import pytimetk as tk

# PARAMETERS ----
INITIAL_SAMPLE_SIZE = 500
TREATMENT_SIZE = 250
ACCEPT_OFFER_PROP = 0.25
ACCEPT_OFFER_AND_NOT_CANCELLED_PROP = 0.75

NEW_LEAD_TIME_MEAN = 65
NEW_LEAD_TIME_STD = 10
CLIP_LOWER = 18
CLIP_UPPER = 99

# DATA CREATION ----
np.random.seed(0) 

hotel_df = pd.read_csv('data/hotel_cancellations_subset.csv')

hotel_df.query('distribution_channel == "Groups"')

hotel_df.glimpse()

hotel_df['market_segment'].value_counts()

groups_df = hotel_df.query('market_segment == "Groups"')

groups_long_lead_df = groups_df.query('lead_time > 180')

groups_long_lead_df.groupby('is_canceled').agg({'is_canceled': 'count'})

groups_long_lead_sample_df = groups_long_lead_df.sample(n=INITIAL_SAMPLE_SIZE, random_state=123)

groups_long_lead_sample_df

treatment_control = np.array(['control'] * (INITIAL_SAMPLE_SIZE - TREATMENT_SIZE) + ['treatment'] * TREATMENT_SIZE)

# Shuffle the array to randomize the order
np.random.shuffle(treatment_control)

treatment_control

groups_long_lead_sample_df['treatment_control'] = treatment_control

df = groups_long_lead_sample_df.copy()

df

# Initialize the 'accept_offer' column with 0s
df['accept_offer'] = 0

# Calculate 25% of the treatment group size
treatment_count = (df['treatment_control'] == 'treatment').sum()
accept_count = int(treatment_count * ACCEPT_OFFER_PROP)

# Randomly select 25% of the treatment group and set their 'accept_offer' value to 1
treatment_indices = df[df['treatment_control'] == 'treatment'].index
accept_indices = np.random.choice(treatment_indices, accept_count, replace=False)
df.loc[accept_indices, 'accept_offer'] = 1

accept_offer_df = df.copy()

accept_offer_df.glimpse()

df = accept_offer_df.copy()

# Copy 'is_cancelled' to 'is_cancelled_2'
df['is_canceled_2'] = df['is_canceled']

# Get indices of those who accepted the offer
accepted_indices = df[df['accept_offer'] == 1].index

# Calculate 75% of those who accepted the offer
not_cancel_count = int(len(accepted_indices) * ACCEPT_OFFER_AND_NOT_CANCELLED_PROP)

# Randomly select 75% of the accepted_indices and set 'is_cancelled_2' to 0
not_cancel_indices = np.random.choice(accepted_indices, not_cancel_count, replace=False)
df.loc[not_cancel_indices, 'is_canceled_2'] = 0

not_canceled_df = df.copy()

not_canceled_df.glimpse()

# New Lead time ----

df = not_canceled_df.copy()

# Initialize new_lead_time with existing lead_time values
df['new_lead_time'] = df['lead_time']

# Get indices of those who accepted the offer
accepted_indices = df[df['accept_offer'] == 1].index

# Generate random numbers with a mean of around 65 and standard deviation that keeps values mostly within 18-99
# Note: Adjust the std_dev as needed to fit your specific distribution requirements
mean_value = NEW_LEAD_TIME_MEAN
std_dev = NEW_LEAD_TIME_STD
random_values = np.random.normal(mean_value, std_dev, len(accepted_indices))

# Clip values to be within 18-99 range
random_values_clipped = np.clip(random_values, CLIP_LOWER, CLIP_UPPER)

# Assign these values to the new_lead_time column for those who accepted the offer
df.loc[accepted_indices, 'new_lead_time'] = random_values_clipped

# Convert to int
df['new_lead_time'] = df['new_lead_time'].astype(int)

new_lead_time_df = df.copy()

new_lead_time_df.glimpse()

# Cleanup ----

df = new_lead_time_df.copy()

df['is_canceled'] = df['is_canceled_2']

df = df.drop(columns=[ 'is_canceled_2'])

cleanup_df = df.copy()

cleanup_df.glimpse()


cleanup_df.to_csv("data/hotel_cancellations_experiment.csv", index=False)

