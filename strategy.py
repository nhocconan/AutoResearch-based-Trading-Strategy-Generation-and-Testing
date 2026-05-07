# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_1dTrend
Hypothesis: Trade breakouts of Camarilla R4/S4 levels only when aligned with daily trend (EMA200) and confirmed by volume spike. 
R4/S4 represent strong breakout levels; daily EMA200 filters counter-trend moves; volume confirms breakout strength. 
Designed for 15-25 trades/year on 6h timeframe to minimize fee drag. Works in bull/bear via daily trend filter.
"""

name = "6h_Camarilla_R4_S4_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla pivot points (using prior day's OHLC)
    # R4 = C + ((H-L) * 1.1/2), S4 = C - ((H-L) * 1.1/2)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    camarilla_range = daily_high - daily_low
    r4 = daily_close + (camarilla_range * 1.1 / 2)
    s4 = daily_close - (camarilla_range * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (with 1-day delay for completed bar)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4, additional_delay_bars=1)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4, additional_delay_bars=1)
    
    # Get daily data for trend filter (EMA200)
    if len(df_1d) < 200:
        return np.zeros(n)
    
    ema_200_1d = pd.Series(daily_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Get 6h volume for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
        if np.isnan(close_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = close_1d_aligned[i] > ema_200_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_200_1d_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above R4 with upward trend and volume spike
            if (close[i] > r4_aligned[i] and 
                trend_up and 
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S4 with downward trend and volume spike
            elif (close[i] < s4_aligned[i] and 
                  trend_down and 
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to daily close or trend turns down
            if close[i] < daily_close[-1] if i == len(prices)-1 else close_1d_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to daily close or trend turns up
            if close[i] > daily_close[-1] if i == len(prices)-1 else close_1d_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals