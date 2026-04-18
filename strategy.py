#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wTrend_Volume
Hypothesis: Camarilla pivot breakouts at R1/S1 levels with weekly EMA trend filter and volume confirmation
capture institutional breakout moves while avoiding false signals in choppy markets. Weekly trend filter
works in both bull and bear markets by aligning with higher timeframe momentum. Target: 15-25 trades/year 
(60-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla pivot levels (using previous day's OHLC)
    # Camarilla formulas: 
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's data
    prev_close = df_1d['close'].shift(1).values  # Previous day's close
    prev_high = df_1d['high'].shift(1).values    # Previous day's high
    prev_low = df_1d['low'].shift(1).values      # Previous day's low
    
    # Calculate Camarilla levels
    R1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    S1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    
    # Align daily levels to 12h timeframe
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    
    # Weekly EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_12h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]) or 
            np.isnan(ema_1w_12h[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_level = R1_12h[i]
        s1_level = S1_12h[i]
        vol_ok = volume_filter[i]
        weekly_trend = ema_1w_12h[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume in uptrend
            if price > r1_level and vol_ok and price > weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume in downtrend
            elif price < s1_level and vol_ok and price < weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Maintain long until price crosses below S1 or trend reverses
            if price < s1_level or price < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Maintain short until price crosses above R1 or trend reverses
            if price > r1_level or price > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0