#!/usr/bin/env python3
name = "6h_ElderRay_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for EMA13, EMA26, and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # Daily EMA13 and EMA26 for Elder Ray calculations
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema26_1d = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA26
    bull_power_1d = high - ema13_1d  # using daily high (not 6h)
    bear_power_1d = low - ema26_1d   # using daily low (not 6h)
    
    # Align Elder Ray components to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Daily trend filter: EMA13 > EMA26 for uptrend, EMA13 < EMA26 for downtrend
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    ema26_aligned = align_htf_to_ltf(prices, df_1d, ema26_1d)
    
    # Volume spike detection (2x 20-period average on 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(26, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(ema13_aligned[i]) or np.isnan(ema26_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: Bull Power > 0 (strength) in daily uptrend with volume
            if bull_power_6h[i] > 0 and ema13_aligned[i] > ema26_aligned[i] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (weakness) in daily downtrend with volume
            elif bear_power_6h[i] < 0 and ema13_aligned[i] < ema26_aligned[i] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power turns negative or trend reverses
            if bull_power_6h[i] <= 0 or ema13_aligned[i] <= ema26_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power turns positive or trend reverses
            if bear_power_6h[i] >= 0 or ema13_aligned[i] >= ema26_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Elder Ray (Bull/Bear Power) with daily trend filter and volume confirmation on 6h
# - Bull Power = High - EMA13; Bear Power = Low - EMA26 (daily)
# - Bull Power > 0 indicates buying strength; Bear Power < 0 indicates selling pressure
# - Trend filter: EMA13 > EMA26 for uptrend, EMA13 < EMA26 for downtrend
# - Volume confirmation (2x average) reduces false signals
# - Works in bull markets (buy strength in uptrend) and bear markets (selling pressure in downtrend)
# - Position size 0.25 targets ~20-40 trades/year to avoid fee drag
# - Uses 1d for Elder Ray and trend, 6h for execution timing
# - Novel application: Elder Ray as trend-following signal with volume confirmation (not commonly tried)