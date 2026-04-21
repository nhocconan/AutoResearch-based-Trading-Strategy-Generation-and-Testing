# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla Pivot Breakout with 12h EMA50 trend filter and volume spike.
Camarilla pivots provide precise support/resistance levels. Breakout above R1 or below S1 with
volume confirmation and aligned trend captures momentum. Uses 12h EMA50 for trend filter to
avoid counter-trend trades. Volume spike (>1.5x average) confirms breakout strength.
Designed for fewer trades (~20-40/year) to minimize fee drag, works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Load 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    rng = high_1d - low_1d
    camarilla_r1 = close_1d + 1.1 * rng / 12
    camarilla_s1 = close_1d - 1.1 * rng / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: volume / 20-period average volume (1d)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_50_12h_aligned[i]
        camarilla_r1 = camarilla_r1_aligned[i]
        camarilla_s1 = camarilla_s1_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        vol_threshold = 1.5  # Volume must be 1.5x average
        
        if position == 0:
            # Enter long: price breaks above R1, volume spike, uptrend
            if (price_close > camarilla_r1 and 
                vol_ratio > vol_threshold and 
                price_close > ema_trend):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1, volume spike, downtrend
            elif (price_close < camarilla_s1 and 
                  vol_ratio > vol_threshold and 
                  price_close < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to pivot point (PP) or trend reversal
            # Calculate pivot point (PP) = (high + low + close)/3
            pp_1d = (high_1d + low_1d + close_1d) / 3
            pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
            pp_val = pp_aligned[i]
            
            if position == 1 and (price_close < pp_val or price_close < ema_trend):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > pp_val or price_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0