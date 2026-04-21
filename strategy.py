#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_R1S1_Breakout_Volume_Filtered_v1
Hypothesis: Use 1d for primary signal direction (Camarilla R1/S1 breakouts), 4h for trend filter (EMA50 slope), and 1h for precise entry timing with volume confirmation.
Trades only in direction of 4h trend to avoid counter-trend whipsaws. Target: 15-30 trades/year (60-120 total over 4 years).
Works in bull/bear by aligning with higher timeframe trend, avoiding false breaks in chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels (signal direction)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1, S1, and pivot point (PP)
    rang = prev_high - prev_low
    r1 = prev_close + 1.1 * rang / 12
    s1 = prev_close - 1.1 * rang / 12
    pp = (prev_high + prev_low + prev_close) / 3
    
    # Align to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Load 4h data once for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # EMA50 for trend direction
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # EMA50 slope (trending up/down)
    ema_slope = np.zeros_like(ema_50_aligned)
    ema_slope[1:] = ema_50_aligned[1:] - ema_50_aligned[:-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(ema_slope[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Trend filter: EMA50 slope positive for long, negative for short
        trending_up = ema_slope[i] > 0
        trending_down = ema_slope[i] < 0
        
        if position == 0:
            # Long conditions: break above R1 + volume + uptrend
            if price > r1_aligned[i] and volume_ok and trending_up:
                signals[i] = 0.20
                position = 1
            # Short conditions: break below S1 + volume + downtrend
            elif price < s1_aligned[i] and volume_ok and trending_down:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below pivot point OR trend turns down
            if price < pp_aligned[i] or ema_slope[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses back above pivot point OR trend turns up
            if price > pp_aligned[i] or ema_slope[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1d_Camarilla_R1S1_Breakout_Volume_Filtered_v1"
timeframe = "1h"
leverage = 1.0