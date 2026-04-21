#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot (R1/S1) breakout with 1d EMA34 trend filter and volume spike.
Camarilla levels (R1/S1) act as short-term support/resistance; breaks above R1 or below S1
with volume confirmation and aligned trend capture momentum. Uses 1d EMA34 for trend filter to
avoid counter-trend trades. Designed for ~25-40 trades/year to minimize fee drag, works in
bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Camarilla pivots and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla pivot levels from previous day (H, L, C)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and ranges for previous day
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1 = close_1d + range_hl * 1.1 / 12
    s1 = close_1d - range_hl * 1.1 / 12
    
    # Align Camarilla levels (use previous day's levels for current day)
    r1_prev = np.roll(r1, 1)
    s1_prev = np.roll(s1, 1)
    r1_prev[0] = np.nan
    s1_prev[0] = np.nan
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_prev)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_prev)
    
    # Volume confirmation: volume / 20-period average volume (1d)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_34_1d_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        vol_threshold = 1.8  # Volume must be 1.8x average to avoid noise
        
        if position == 0:
            # Enter long: price breaks above R1, volume spike, uptrend
            if (price_close > r1_val and 
                vol_ratio > vol_threshold and 
                price_close > ema_trend):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1, volume spike, downtrend
            elif (price_close < s1_val and 
                  vol_ratio > vol_threshold and 
                  price_close < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to pivot or trend reversal
            pivot_val = (high_1d[i] + low_1d[i] + close_1d[i]) / 3
            pivot_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(close_1d, pivot_val))[i]
            
            if position == 1 and (price_close < pivot_aligned or price_close < ema_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > pivot_aligned or price_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0