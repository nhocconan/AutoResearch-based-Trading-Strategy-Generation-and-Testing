#!/usr/bin/env python3
# 1h_4h_Camarilla_R1_S1_Breakout_Volume_Session
# Hypothesis: 1h breakouts of 4h Camarilla R1 (resistance) and S1 (support) levels with volume confirmation and session filter (08-20 UTC).
# Uses 4h trend filter (EMA34) to avoid counter-trend trades. Target: 15-30 trades/year per symbol to avoid fee drag.
# Works in bull/bear via trend filter and session focus on active liquidity hours.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_Camarilla_R1_S1_Breakout_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data ONCE before loop for pivots and EMA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # === Calculate 4h Camarilla R1, S1 levels ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Pivot point and range
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    
    # Camarilla R1 = close + (range * 1.1/2), S1 = close - (range * 1.1/2)
    r1_4h = close_4h + (range_4h * 1.1 / 2)
    s1_4h = close_4h - (range_4h * 1.1 / 2)
    
    # === 4h EMA34 for trend filter ===
    close_4h_series = pd.Series(close_4h)
    ema34_4h = close_4h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 1h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # Pre-compute session hours
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA and volume MA warmup
        # Skip if outside trading session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        r1_4h_val = r1_4h[i]
        s1_4h_val = s1_4h[i]
        ema34_4h_val = ema34_4h[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_4h_val) or np.isnan(s1_4h_val) or np.isnan(ema34_4h_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 4h R1 with volume confirmation and above 4h EMA34
            if (close_val > r1_4h_val and vol_ratio_val > 2.0 and 
                close_val > ema34_4h_val):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 4h S1 with volume confirmation and below 4h EMA34
            elif (close_val < s1_4h_val and vol_ratio_val > 2.0 and 
                  close_val < ema34_4h_val):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below 4h S1
            if close_val <= s1_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Price returns to or above 4h R1
            if close_val >= r1_4h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals