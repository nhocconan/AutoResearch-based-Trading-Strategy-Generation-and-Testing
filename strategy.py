# 6h_Donchian_WeeklyPivot_Trend
# Donchian breakout with weekly pivot direction filter on 6h.
# Long when price breaks above Donchian(20) upper and weekly pivot indicates bullish bias.
# Short when price breaks below Donchian(20) lower and weekly pivot indicates bearish bias.
# Volume confirmation required to filter weak breakouts.
# Designed for ~20-40 trades per year to minimize fee drag.

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
    
    # Donchian Channel parameters (20-period)
    donch_period = 20
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(donch_period, n):
        donch_high[i] = np.max(high[i-donch_period+1:i+1])
        donch_low[i] = np.min(low[i-donch_period+1:i+1])
    
    # Load weekly data for pivot points (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard floor pivot)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Warmup for Donchian and weekly alignment
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high with bullish weekly bias and volume spike
            if (close[i] > donch_high[i] and 
                close[i] > pivot_aligned[i] and 
                close[i] > s1_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with bearish weekly bias and volume spike
            elif (close[i] < donch_low[i] and 
                  close[i] < pivot_aligned[i] and 
                  close[i] < r1_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below weekly pivot or Donchian low
            if close[i] < pivot_aligned[i] or close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above weekly pivot or Donchian high
            if close[i] > pivot_aligned[i] or close[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_WeeklyPivot_Trend"
timeframe = "6h"
leverage = 1.0