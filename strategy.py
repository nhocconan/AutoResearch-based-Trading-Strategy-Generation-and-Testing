#!/usr/bin/env python3
"""
1h_Pivot_R1S1_Breakout_4hTrend_1dVolume
Hypothesis: Use 1-hour Camarilla pivot points (R1/S1) for entry timing with 4-hour trend filter (close > EMA20) and 1-day volume confirmation (volume > 1.5x 20-day average) to capture institutional breakouts in both bull and bear markets. The 4h EMA20 filters trend direction, reducing false breakouts in sideways markets. Target: 15-30 trades/year (60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-hour pivot points (using previous bar's high/low/close)
    pivot = (np.roll(high, 1) + np.roll(low, 1) + np.roll(close, 1)) / 3
    r1 = 2 * pivot - np.roll(low, 1)
    s1 = 2 * pivot - np.roll(high, 1)
    # Set first value to NaN (no previous bar)
    pivot[0] = np.nan
    r1[0] = np.nan
    s1[0] = np.nan
    
    # 4-hour EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1-day volume filter
    df_1d = get_htf_data(prices, '1d')
    vol_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pivot[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = volume[i] > (1.5 * vol_1d_aligned[i])
        ema_trend = ema_4h_aligned[i]
        
        if position == 0:
            # Long: break above R1 with volume in uptrend
            if price > r1[i] and vol_ok and price > ema_trend:
                signals[i] = 0.20
                position = 1
            # Short: break below S1 with volume in downtrend
            elif price < s1[i] and vol_ok and price < ema_trend:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Maintain long until price breaks below S1 or trend reverses
            if price < s1[i] or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Maintain short until price breaks above R1 or trend reverses
            if price > r1[i] or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Pivot_R1S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0