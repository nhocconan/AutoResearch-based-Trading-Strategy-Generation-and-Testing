#!/usr/bin/env python3
"""
1d_Weekly_Camarilla_R1_S1_Breakout_MonthlyTrend_VolumeSpike
Hypothesis: Breakouts from monthly (4-week) Camarilla R1/S1 levels with weekly trend filter and volume spike confirmation.
Monthly timeframe provides strong trend filter that works in both bull and bear markets.
Volume spike (>2x 30-period average) confirms breakout strength. Designed for low trade frequency (~10-30/year)
to minimize fee drag. Uses 1d timeframe for execution with weekly trend filter.
"""

name = "1d_Weekly_Camarilla_R1_S1_Breakout_MonthlyTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for monthly (4-week) calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 4:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 4-week (monthly) high, low, close for Camarilla
    # Using 4-week lookback for monthly levels
    monthly_high = np.full_like(close_1w, np.nan)
    monthly_low = np.full_like(close_1w, np.nan)
    monthly_close = np.full_like(close_1w, np.nan)
    
    if len(high_1w) >= 4:
        for i in range(3, len(high_1w)):
            monthly_high[i] = np.max(high_1w[i-3:i+1])
            monthly_low[i] = np.min(low_1w[i-3:i+1])
            monthly_close[i] = close_1w[i]  # Current week's close
    
    # Previous month's values for Camarilla calculation (4-week lag)
    pmh = np.concatenate([[monthly_high[0]] * 3, monthly_high[:-3]])  # previous month high
    pml = np.concatenate([[monthly_low[0]] * 3, monthly_low[:-3]])    # previous month low
    pmc = np.concatenate([[monthly_close[0]] * 3, monthly_close[:-3]]) # previous month close
    
    # Calculate monthly Camarilla levels (R1, S1 are the key breakout levels)
    rang = pmh - pml
    r1 = pmc + 1.1 * rang * 1.0833  # R1 = Close + 1.1 * (High-Low) * 1.0833
    s1 = pmc - 1.1 * rang * 1.0833  # S1 = Close - 1.1 * (High-Low) * 1.0833
    
    # Align monthly Camarilla levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[0:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (ema_20_1w[i-1] * 19 + close_1w[i]) / 20
    
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume spike filter: current volume / 30-day average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 30:
        vol_ma[29] = np.mean(volume[0:30])
        for i in range(30, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 29 + volume[i]) / 30
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(30, 20)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: price breaks above R1 AND uptrend (price > EMA20) AND volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_20_1w_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: price breaks below S1 AND downtrend (price < EMA20) AND volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 5 days
            if bars_since_entry < 5:
                signals[i] = 0.25
            else:
                # Exit long: price breaks below S1 OR trend reversal (price < EMA20)
                if close[i] < s1_aligned[i] or close[i] < ema_20_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 5 days
            if bars_since_entry < 5:
                signals[i] = -0.25
            else:
                # Exit short: price breaks above R1 OR trend reversal (price > EMA20)
                if close[i] > r1_aligned[i] or close[i] > ema_20_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals