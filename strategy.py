#!/usr/bin/env python3
"""
12h_12hr_Camarilla_R1S1_Breakout_1wTrend_Volume
Hypothesis: Camarilla R1/S1 levels from daily + weekly trend filter + volume confirmation 
provides high-probability breakout trades with low frequency. Target: 12-37 trades/year.
Works in bull/bear via trend filter and volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day (use 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla R1 and S1 levels
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to 12h timeframe
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    
    # Weekly trend filter (EMA 34)
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
    
    start_idx = 30  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]) or 
            np.isnan(ema_1w_12h[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = R1_12h[i]
        s1 = S1_12h[i]
        vol_ok = volume_filter[i]
        weekly_trend = ema_1w_12h[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume in uptrend
            if price > r1 and vol_ok and price > weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume in downtrend
            elif price < s1 and vol_ok and price < weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price breaks below S1 or trend reverses
            if price < s1 or price < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price breaks above R1 or trend reverses
            if price > r1 or price > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_12hr_Camarilla_R1S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0