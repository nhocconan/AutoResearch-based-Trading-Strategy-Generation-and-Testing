#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h/1d Camarilla R1/S1 breakout with volume confirmation and session filter
# Uses 4h for trend direction (price > 4h EMA50 = uptrend, < = downtrend)
# 1d for Camarilla pivot levels (R1, S1) from prior completed 1d bar
# 1h for entry timing: break R1 with volume spike = long, break S1 with volume spike = short
# Session filter: 08-20 UTC to avoid low-liquidity hours
# Discrete sizing 0.20 targets 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Volume confirmation: 1.5x 20-period EMA of volume reduces false breakouts

name = "1h_Camarilla_R1S1_4hEMA50_1dPivot_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 4h EMA50 trend filter from prior completed 4h bar
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_shifted = np.roll(ema50_4h, 1)
    ema50_4h_shifted[0] = np.nan
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h_shifted)
    
    # Calculate 1d Camarilla pivot levels (R1, S1) from prior completed 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior completed 1d bar values (shift by 1)
    if len(high_1d) < 2:
        return np.zeros(n)
    phigh = np.roll(high_1d, 1)[-1]  # prior day high
    plow = np.roll(low_1d, 1)[-1]    # prior day low
    pclose = np.roll(close_1d, 1)[-1] # prior day close
    
    # Camarilla equations for R1 and S1
    rang = phigh - plow
    r1 = pclose + (rang * 1.1 / 12)
    s1 = pclose - (rang * 1.1 / 12)
    
    # Create arrays of R1, S1 for each 1d bar
    r1_1d = np.full_like(close_1d, r1)
    s1_1d = np.full_like(close_1d, s1)
    
    # Align Camarilla levels to 1h timeframe (wait for 1d bar to close)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(vol_ema_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R1 AND 4h uptrend AND volume spike
            if close[i] > r1_1d_aligned[i] and close[i] > ema50_4h_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S1 AND 4h downtrend AND volume spike
            elif close[i] < s1_1d_aligned[i] and close[i] < ema50_4h_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below S1 OR 4h trend turns down
            if close[i] < s1_1d_aligned[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above R1 OR 4h trend turns up
            if close[i] > r1_1d_aligned[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals