#!/usr/bin/env python3
"""
1d_1w_Weekly_High_Low_Breakout_Volume
Hypothesis: Break above weekly high or below weekly low with volume confirmation.
Weekly levels act as strong support/resistance. Works in bull (breakouts) and bear (fading false breaks)
by requiring volume spike and using weekly context. Targets 10-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Weekly_High_Low_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for high/low levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly high and low
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Align weekly levels to daily
    weekly_high_d = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_d = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Volume average (20-day) for confirmation
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(weekly_high_d[i]) or np.isnan(weekly_low_d[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 2x average
        vol_confirm = volume[i] > vol_ma[i] * 2.0
        
        # Breakout conditions
        breakout_up = high[i] > weekly_high_d[i] and vol_confirm
        breakout_down = low[i] < weekly_low_d[i] and vol_confirm
        
        # Entry logic
        long_entry = breakout_up
        short_entry = breakout_down
        
        # Exit logic: opposite breakout or price returns to weekly midpoint
        weekly_mid = (weekly_high_d[i] + weekly_low_d[i]) / 2
        long_exit = breakout_down or close[i] < weekly_mid
        short_exit = breakout_up or close[i] > weekly_mid
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals