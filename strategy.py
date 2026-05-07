#!/usr/bin/env python3
name = "4h_Pivots_Trend_Scalp"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Pivot points from previous day (standard formula)
    c_high = df_1d['high'].values
    c_low = df_1d['low'].values
    c_close = df_1d['close'].values
    
    pivot = (c_high + c_low + c_close) / 3
    range_val = c_high - c_low
    r1 = pivot + (range_val * 1.0)  # R1: P + 1*R
    s1 = pivot - (range_val * 1.0)  # S1: P - 1*R
    r2 = pivot + (range_val * 2.0)  # R2: P + 2*R
    s2 = pivot - (range_val * 2.0)  # S2: P - 2*R
    
    # Align pivot levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(c_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(r2_4h[i]) or np.isnan(s2_4h[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: price above R1 and rising in daily uptrend with volume
            if close[i] > r1_4h[i] and ema_34_4h[i] > ema_34_4h[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price below S1 and falling in daily downtrend with volume
            elif close[i] < s1_4h[i] and ema_34_4h[i] < ema_34_4h[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to S1 or trend reverses
            if close[i] < s1_4h[i] or ema_34_4h[i] < ema_34_4h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to R1 or trend reverses
            if close[i] > r1_4h[i] or ema_34_4h[i] > ema_34_4h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Pivot point (R1/S1) breakouts with daily trend filter and volume confirmation
# - Uses standard pivot points (R1/S1) from previous day as support/resistance
# - Breakout above R1 in daily uptrend (EMA34 rising) signals bullish continuation
# - Breakdown below S1 in daily downtrend (EMA34 falling) signals bearish continuation
# - Volume confirmation (1.5x average) reduces false breakouts
# - Exit when price returns to S1/R1 or daily trend reverses
# - Position size 0.25 targets ~25-35 trades/year to avoid fee drag
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Uses 1d timeframe for structure and trend, 4h for execution timing
# - Pivot points are more widely used than Camarilla, potentially less crowded
# - Simpler than Camarilla (only R1/S1) to reduce overfitting and increase robustness