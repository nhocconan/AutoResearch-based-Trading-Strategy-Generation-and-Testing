#!/usr/bin/env python3
# 6h_WeeklyPivot_DonchianBreakout_Volume
# Hypothesis: On 6h timeframe, enter long when price breaks above weekly Donchian high (20-period) 
# with price above weekly pivot and volume confirmation. Enter short when price breaks below 
# weekly Donchian low with price below weekly pivot and volume confirmation. Exit when price 
# crosses weekly pivot (trend reversal). Uses weekly pivot for trend filter and Donchian 
# breakout for momentum, aiming to capture trends in both bull and bear markets with low 
# frequency to minimize fee drag.

name = "6h_WeeklyPivot_DonchianBreakout_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot calculation and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly pivot point (standard: (H+L+C)/3)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Calculate weekly Donchian channels (20-period high/low)
    # Highest high of last 20 weekly bars
    donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 weekly bars
    donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly data to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Volume confirmation: 20-period moving average on 6x timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        pivot_val = weekly_pivot_aligned[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price breaks above weekly Donchian high with price > weekly pivot and volume > 20MA
            if close[i] > donchian_high_val and close[i] > pivot_val and volume[i] > vol_ma_val:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly Donchian low with price < weekly pivot and volume > 20MA
            elif close[i] < donchian_low_val and close[i] < pivot_val and volume[i] > vol_ma_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below weekly pivot (trend reversal)
            if close[i] < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above weekly pivot (trend reversal)
            if close[i] > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals