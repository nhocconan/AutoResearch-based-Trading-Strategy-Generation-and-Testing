#!/usr/bin/env python3
# 1d_1w_Camarilla_Pivot_Breakout_v3
# Hypothesis: Use weekly Camarilla pivot levels on 1d timeframe with volume confirmation.
# Long when price breaks above weekly R4 with volume > 1.3x 20-period average,
# short when breaks below weekly S4 with volume > 1.3x 20-period average.
# Weekly Camarilla levels provide strong institutional support/resistance.
# Volume confirms breakout strength to avoid false signals.
# Designed for low trade frequency (target: 30-100 total over 4 years) to minimize fee drag.
# Works in bull via breakouts above resistance, in bear via breakdowns below support.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Pivot_Breakout_v3"
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
    
    # Weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for Camarilla calculation
    prev_high = df_1w['high'].iloc[-2] if len(df_1w) >= 2 else df_1w['high'].iloc[-1]
    prev_low = df_1w['low'].iloc[-2] if len(df_1w) >= 2 else df_1w['low'].iloc[-1]
    prev_close = df_1w['close'].iloc[-2] if len(df_1w) >= 2 else df_1w['close'].iloc[-1]
    
    # Calculate weekly Camarilla pivot levels
    range_val = prev_high - prev_low
    if range_val <= 0:
        return np.zeros(n)
    
    # Weekly R4 and S4 levels (Camarilla formula)
    weekly_r4 = prev_close + range_val * 1.1 / 2  # R4 = Close + (Range * 1.1/2)
    weekly_s4 = prev_close - range_val * 1.1 / 2  # S4 = Close - (Range * 1.1/2)
    
    # Align weekly levels to 1d timeframe
    weekly_r4_array = np.full(len(df_1w), weekly_r4)
    weekly_s4_array = np.full(len(df_1w), weekly_s4)
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1w, weekly_r4_array)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1w, weekly_s4_array)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values  # default to 1.0 if no MA
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(weekly_r4_aligned[i]) or np.isnan(weekly_s4_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with volume filter
        long_breakout = close[i] > weekly_r4_aligned[i] and vol_ratio[i] > 1.3
        short_breakout = close[i] < weekly_s4_aligned[i] and vol_ratio[i] > 1.3
        
        # Exit conditions: return to weekly Camarilla pivot level (R3/S3)
        weekly_pivot = (prev_high + prev_low + prev_close) / 3
        weekly_r3 = prev_close + range_val * 1.1 / 4  # R3 = Close + (Range * 1.1/4)
        weekly_s3 = prev_close - range_val * 1.1 / 4  # S3 = Close - (Range * 1.1/4)
        
        weekly_r3_array = np.full(len(df_1w), weekly_r3)
        weekly_s3_array = np.full(len(df_1w), weekly_s3)
        weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3_array)
        weekly_s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3_array)
        
        long_exit = close[i] < weekly_r3_aligned[i]
        short_exit = close[i] > weekly_s3_aligned[i]
        
        # Signal logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
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