# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:07:20 2025

@author: Ryan.Larson
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from statsmodels.stats.power import TTestPower

def add_good_range_patch(ax, min_acceptable_value, target_value):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    accept_rect = patches.Rectangle(
        xy=(-0.5, min_acceptable_value),  # lower left corner of box: beginning of x-axis range & y coord)
        width=9.0,  # width from x-axis range
        # width=ax.get_xlim()[1]-ax.get_xlim()[0],  # width from x-axis range
        height= target_value - min_acceptable_value,
        color='green', alpha=0.1, ec='green', zorder=3
    )
    target_rect = patches.Rectangle(
        xy=(-0.5, target_value),  # lower left corner of box: beginning of x-axis range & y coord)
        width=9.0,  # width from x-axis range
        # width=ax.get_xlim()[1]-ax.get_xlim()[0],  # width from x-axis range
        height= ymax - target_value,
        color='green', alpha=0.3, ec='green', zorder=2
    )
    bad_rect = patches.Rectangle(
        xy=(-0.5, ax.get_ylim()[0]),  # lower left corner of box: beginning of x-axis range & y coord)
        width=9.0,  # width from x-axis range
        # width=ax.get_xlim()[1]-ax.get_xlim()[0],  # width from x-axis range
        height= min_acceptable_value - ymin,
        color='red', alpha=0.1, ec='red', zorder=1
    )
    
    ax.add_patch(accept_rect)
    ax.add_patch(target_rect)
    ax.add_patch(bad_rect)


filename = "C:/Users/Ryan.Larson.ROCKWELLINC/github/mark10-processing/Material Testing Unified Results - Raw Data.csv"
df = pd.read_csv(filename)

df_tensile = df[df['Material Test Type (Flexural/Tensile)']=='T'].copy()
df_flex_h = df[(df['Material Test Type (Flexural/Tensile)']=='F') & (df['Orientation (V/H)']=='H')].copy()
df_flex_v = df[(df['Material Test Type (Flexural/Tensile)']=='F') & (df['Orientation (V/H)']=='V')].copy()

sides = ['P1','P3','P4','P6']
center = ['P2','P5']

# # # Tensile Properties
# # # plt.figure(dpi=300)
# # # sns.scatterplot(data=df_tensile[df_tensile['Position Number'].isin(sides)], x='Modulus (Mpa)', y='Strength (Mpa)', hue='Truck Number')
# # # plt.title('Tensile Material Properties - Sides')
# # # plt.ylim([40,75])
# # # plt.legend(loc="lower right")

# # # plt.figure(dpi=300)
# # # sns.scatterplot(data=df_tensile[df_tensile['Position Number'].isin(center)], x='Modulus (Mpa)', y='Strength (Mpa)', hue='Truck Number')
# # # plt.title('Tensile Material Properties - Center')
# # # plt.legend(loc="lower right")

# # plt.figure(dpi=300)
# # sns.boxplot(data=df_tensile[df_tensile['Position Number'].isin(sides)], x='Truck Number', y='Modulus (Mpa)', hue='Truck Number')
# # plt.title('Tensile Modulus - Sides')

# # plt.figure(dpi=300)
# # sns.boxplot(data=df_tensile[df_tensile['Position Number'].isin(center)], x='Truck Number', y='Modulus (Mpa)', hue='Truck Number')
# # plt.title('Tensile Modulus - Center')

# # plt.figure(dpi=300)
# # sns.boxplot(data=df_tensile[df_tensile['Position Number'].isin(sides)], x='Truck Number', y='Strength (Mpa)', hue='Truck Number')
# # plt.title('Tensile Strength - Sides')

# # plt.figure(dpi=300)
# # sns.boxplot(data=df_tensile[df_tensile['Position Number'].isin(center)], x='Truck Number', y='Strength (Mpa)', hue='Truck Number')
# # plt.title('Tensile Strength - Center')

# # # Flexural V Properties
# # # plt.figure(dpi=300)
# # # sns.scatterplot(data=df_flex_v[df_flex_v['Position Number'].isin(sides)], x='Modulus (Mpa)', y='Strength (Mpa)', hue='Truck Number')
# # # plt.title('Flexural V Material Properties - Sides')
# # # plt.xlim([-500,10000])
# # # plt.ylim([0,150])
# # # plt.legend(loc="lower right")

# # # plt.figure(dpi=300)
# # # sns.scatterplot(data=df_flex_v[df_flex_v['Position Number'].isin(center)], x='Modulus (Mpa)', y='Strength (Mpa)', hue='Truck Number')
# # # plt.title('Flexural V Material Properties - Center')
# # # plt.legend(loc="lower right")

# # # plt.figure(dpi=300)
# # # sns.scatterplot(data=df_flex_v[df_flex_v['Position Number'].isin(sides)], x='Specimen Thickness (mm)', y='Modulus (Mpa)', hue='Truck Number')
# # # plt.ylim([-500,10000])
# # # plt.title('Flex V Modulus vs Sample Thickness - Sides')

# # # plt.figure(dpi=300)
# # # sns.scatterplot(data=df_flex_v[df_flex_v['Position Number'].isin(sides)], x='Specimen Thickness (mm)', y='Strength (Mpa)', hue='Truck Number')
# # # plt.ylim([0,150])
# # # plt.title('Flex V Strength vs Sample Thickness - Sides')

# # plt.figure(dpi=300)
# # sns.boxplot(data=df_flex_v[df_flex_v['Position Number'].isin(sides)], x='Truck Number', y='Modulus (Mpa)', hue='Truck Number')
# # plt.title('Flex V Modulus - Sides')
# # plt.ylim([-500,10000])

# # plt.figure(dpi=300)
# # sns.boxplot(data=df_flex_v[df_flex_v['Position Number'].isin(center)], x='Truck Number', y='Modulus (Mpa)', hue='Truck Number')
# # plt.title('Flex V Modulus - Center')

# # plt.figure(dpi=300)
# # sns.boxplot(data=df_flex_v[df_flex_v['Position Number'].isin(sides)], x='Truck Number', y='Strength (Mpa)', hue='Truck Number')
# # plt.title('Flex V Strength - Sides')

# # plt.figure(dpi=300)
# # sns.boxplot(data=df_flex_v[df_flex_v['Position Number'].isin(center)], x='Truck Number', y='Strength (Mpa)', hue='Truck Number')
# # plt.title('Flex V Strength - Center')

# # # Flexural H Properties
# # # plt.figure(dpi=300)
# # # sns.scatterplot(data=df_flex_h[df_flex_h['Position Number'].isin(sides)], x='Modulus (Mpa)', y='Strength (Mpa)', hue='Truck Number')
# # # plt.title('Flexural H Material Properties - Sides')
# # # # plt.xlim([-500,10000])
# # # # plt.ylim([0,150])
# # # plt.legend(loc="lower right")

# # # plt.figure(dpi=300)
# # # sns.scatterplot(data=df_flex_h[df_flex_h['Position Number'].isin(center)], x='Modulus (Mpa)', y='Strength (Mpa)', hue='Truck Number')
# # # plt.title('Flexural H Material Properties - Center')
# # # plt.legend(loc="lower right")

# filename = "C:/Users/Ryan.Larson.ROCKWELLINC/github/mark10-processing/Material Testing Unified Results - Wall Pull Off.csv"
# df = pd.read_csv(filename)

# fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
# sns.boxplot(data=df, x='Truck Number', y='Wt Avg Tensile Modulus - V (MPa)', hue='Truck Number', zorder=4)
# add_good_range_patch(ax, 4250, 4750)
# plt.title('Weighted Average Tensile Modulus')

# fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
# sns.boxplot(data=df, x='Truck Number', y='Wt Avg Tensile Strength - V (MPa)', hue='Truck Number', zorder=4)
# add_good_range_patch(ax, 55, 68)
# plt.title('Weighted Average Tensile Strength')

# fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
# sns.boxplot(data=df, x='Truck Number', y='Wt Avg Flexural Modulus - V (MPa)', hue='Truck Number', zorder=4)
# add_good_range_patch(ax, 5200, 6400)
# plt.title('Weighted Average Flexural Modulus - V')

# fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
# sns.boxplot(data=df, x='Truck Number', y='Wt Avg Flexural Strength - V (MPa)', hue='Truck Number', zorder=4)
# add_good_range_patch(ax, 86, 93)
# plt.title('Weighted Average Flexural Strength - V')

# fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
# sns.boxplot(data=df, x='Truck Number', y='Wt Avg Flexural Modulus - H (MPa)', hue='Truck Number', zorder=4)
# add_good_range_patch(ax, 1930, 2145)
# plt.title('Weighted Average Flexural Modulus - H')

# fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
# sns.boxplot(data=df, x='Truck Number', y='Wt Avg Flexural Strength - H (MPa)', hue='Truck Number', zorder=4)
# add_good_range_patch(ax, 40, 44)
# plt.title('Weighted Average Flexural Strength - H')


# Calculating expected effect sizes for UV paired testing

df_tensile_side = df_tensile[df_tensile['Position is Side (Bool)']==True]
df_tensile_center = df_tensile[df_tensile['Position is Side (Bool)']==False]

df_flex_h_side = df_flex_h[df_flex_h['Position is Side (Bool)']==True]
df_flex_h_center = df_flex_h[df_flex_h['Position is Side (Bool)']==False]

df_flex_v_side = df_flex_v[df_flex_v['Position is Side (Bool)']==True]
df_flex_v_center = df_flex_v[df_flex_v['Position is Side (Bool)']==False]

tens_str_factor = 0.96
tens_mod_factor = 3
flex_str_factor = 0.96
flex_mod_factor = 0.99

# Tensile
noise = np.random.normal(loc=0, scale=2, size=df_tensile_side['Strength (Mpa)'].shape)  # Adjust scale to match expected variability
tens_side_diff_str_mean = (df_tensile_side['Strength (Mpa)'] - tens_str_factor*df_tensile_side['Strength (Mpa)'] + noise).mean()
tens_side_diff_str_std = (df_tensile_side['Strength (Mpa)'] - tens_str_factor*df_tensile_side['Strength (Mpa)'] + noise).std()
effect_size = tens_side_diff_str_mean / tens_side_diff_str_std
power_analysis = TTestPower()
sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.8, alternative='two-sided')
print(f'Tensile Strength Side Sample Size:\t{sample_size:.1f}')

noise = np.random.normal(loc=0, scale=2, size=df_tensile_side['Modulus (Mpa)'].shape)  # Adjust scale to match expected variability
tens_side_diff_mod_mean = (df_tensile_side['Modulus (Mpa)'] - tens_str_factor*df_tensile_side['Modulus (Mpa)'] + noise).mean()
tens_side_diff_mod_std = (df_tensile_side['Modulus (Mpa)'] - tens_str_factor*df_tensile_side['Modulus (Mpa)'] + noise).std()
effect_size = tens_side_diff_mod_mean / tens_side_diff_mod_std
power_analysis = TTestPower()
sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.8, alternative='two-sided')
print(f'Tensile Modulus Side Sample Size:\t{sample_size:.1f}')

noise = np.random.normal(loc=0, scale=2, size=df_tensile_center['Strength (Mpa)'].shape)  # Adjust scale to match expected variability
tens_center_diff_str_mean = (df_tensile_center['Strength (Mpa)'] - tens_str_factor*df_tensile_center['Strength (Mpa)'] + noise).mean()
tens_center_diff_str_std = (df_tensile_center['Strength (Mpa)'] - tens_str_factor*df_tensile_center['Strength (Mpa)'] + noise).std()
effect_size = tens_center_diff_str_mean / tens_center_diff_str_std
power_analysis = TTestPower()
sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.8, alternative='two-sided')
print(f'Tensile Strength Center Sample Size:\t{sample_size:.1f}')

noise = np.random.normal(loc=0, scale=2, size=df_tensile_center['Modulus (Mpa)'].shape)  # Adjust scale to match expected variability
tens_center_diff_mod_mean = (df_tensile_center['Modulus (Mpa)'] - tens_str_factor*df_tensile_center['Modulus (Mpa)'] + noise).mean()
tens_center_diff_mod_std = (df_tensile_center['Modulus (Mpa)'] - tens_str_factor*df_tensile_center['Modulus (Mpa)'] + noise).std()
effect_size = tens_center_diff_mod_mean / tens_center_diff_mod_std
power_analysis = TTestPower()
sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.8, alternative='two-sided')
print(f'Tensile Modulus Center Sample Size:\t{sample_size:.1f}')

# Flex H
noise = np.random.normal(loc=0, scale=2, size=df_flex_h_side['Strength (Mpa)'].shape)  # Adjust scale to match expected variability
flex_h_side_diff_str_mean = (df_flex_h_side['Strength (Mpa)'] - flex_str_factor*df_flex_h_side['Strength (Mpa)'] + noise).mean()
flex_h_side_diff_str_std = (df_flex_h_side['Strength (Mpa)'] - flex_str_factor*df_flex_h_side['Strength (Mpa)'] + noise).std()
effect_size = flex_h_side_diff_str_mean / flex_h_side_diff_str_std
power_analysis = TTestPower()
sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.8, alternative='two-sided')
print(f'\nFlex H Strength Side Sample Size:\t{sample_size:.1f}')

noise = np.random.normal(loc=0, scale=2, size=df_flex_h_side['Modulus (Mpa)'].shape)  # Adjust scale to match expected variability
tens_side_diff_mod_mean = (df_flex_h_side['Modulus (Mpa)'] - flex_mod_factor*df_flex_h_side['Modulus (Mpa)'] + noise).mean()
tens_side_diff_mod_std = (df_flex_h_side['Modulus (Mpa)'] - flex_mod_factor*df_flex_h_side['Modulus (Mpa)'] + noise).std()
effect_size = tens_side_diff_mod_mean / tens_side_diff_mod_std
power_analysis = TTestPower()
sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.8, alternative='two-sided')
print(f'Flex H Modulus Side Sample Size:\t{sample_size:.1f}')

noise = np.random.normal(loc=0, scale=2, size=df_flex_h_center['Strength (Mpa)'].shape)  # Adjust scale to match expected variability
flex_h_center_diff_str_mean = (df_flex_h_center['Strength (Mpa)'] - flex_str_factor*df_flex_h_center['Strength (Mpa)'] + noise).mean()
flex_h_center_diff_str_std = (df_flex_h_center['Strength (Mpa)'] - flex_str_factor*df_flex_h_center['Strength (Mpa)'] + noise).std()
effect_size = flex_h_center_diff_str_mean / flex_h_center_diff_str_std
power_analysis = TTestPower()
sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.8, alternative='two-sided')
print(f'Flex H Strength Center Sample Size:\t{sample_size:.1f}')

noise = np.random.normal(loc=0, scale=2, size=df_flex_h_center['Modulus (Mpa)'].shape)  # Adjust scale to match expected variability
tens_center_diff_mod_mean = (df_flex_h_center['Modulus (Mpa)'] - flex_mod_factor*df_flex_h_center['Modulus (Mpa)'] + noise).mean()
tens_center_diff_mod_std = (df_flex_h_center['Modulus (Mpa)'] - flex_mod_factor*df_flex_h_center['Modulus (Mpa)'] + noise).std()
effect_size = tens_center_diff_mod_mean / tens_center_diff_mod_std
power_analysis = TTestPower()
sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.8, alternative='two-sided')
print(f'Flex H Modulus Center Sample Size:\t{sample_size:.1f}')

# Flex V
noise = np.random.normal(loc=0, scale=2, size=df_flex_v_side['Strength (Mpa)'].shape)  # Adjust scale to match expected variability
flex_v_side_diff_str_mean = (df_flex_v_side['Strength (Mpa)'] - flex_str_factor*df_flex_v_side['Strength (Mpa)'] + noise).mean()
flex_v_side_diff_str_std = (df_flex_v_side['Strength (Mpa)'] - flex_str_factor*df_flex_v_side['Strength (Mpa)'] + noise).std()
effect_size = flex_v_side_diff_str_mean / flex_v_side_diff_str_std
power_analysis = TTestPower()
sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.8, alternative='two-sided')
print(f'\nFlex V Strength Side Sample Size:\t{sample_size:.1f}')

noise = np.random.normal(loc=0, scale=2, size=df_flex_v_side['Modulus (Mpa)'].shape)  # Adjust scale to match expected variability
tens_side_diff_mod_mean = (df_flex_v_side['Modulus (Mpa)'] - flex_mod_factor*df_flex_v_side['Modulus (Mpa)'] + noise).mean()
tens_side_diff_mod_std = (df_flex_v_side['Modulus (Mpa)'] - flex_mod_factor*df_flex_v_side['Modulus (Mpa)'] + noise).std()
effect_size = tens_side_diff_mod_mean / tens_side_diff_mod_std
power_analysis = TTestPower()
sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.8, alternative='two-sided')
print(f'Flex V Modulus Side Sample Size:\t{sample_size:.1f}')

noise = np.random.normal(loc=0, scale=2, size=df_flex_v_center['Strength (Mpa)'].shape)  # Adjust scale to match expected variability
flex_v_center_diff_str_mean = (df_flex_v_center['Strength (Mpa)'] - flex_str_factor*df_flex_v_center['Strength (Mpa)'] + noise).mean()
flex_v_center_diff_str_std = (df_flex_v_center['Strength (Mpa)'] - flex_str_factor*df_flex_v_center['Strength (Mpa)'] + noise).std()
effect_size = flex_v_center_diff_str_mean / flex_v_center_diff_str_std
power_analysis = TTestPower()
sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.8, alternative='two-sided')
print(f'Flex V Strength Center Sample Size:\t{sample_size:.1f}')

noise = np.random.normal(loc=0, scale=2, size=df_flex_v_center['Modulus (Mpa)'].shape)  # Adjust scale to match expected variability
tens_center_diff_mod_mean = (df_flex_v_center['Modulus (Mpa)'] - flex_mod_factor*df_flex_v_center['Modulus (Mpa)'] + noise).mean()
tens_center_diff_mod_std = (df_flex_v_center['Modulus (Mpa)'] - flex_mod_factor*df_flex_v_center['Modulus (Mpa)'] + noise).std()
effect_size = tens_center_diff_mod_mean / tens_center_diff_mod_std
power_analysis = TTestPower()
sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=0.05, power=0.8, alternative='two-sided')
print(f'Flex V Modulus Center Sample Size:\t{sample_size:.1f}')
