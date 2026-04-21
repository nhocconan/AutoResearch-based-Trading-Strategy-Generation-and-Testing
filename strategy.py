#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R + 1d EMA trend + volume spike for mean-reversion entries.
Williams %R identifies overbought/oversold conditions on 4h. In ranging markets (common in 2025),
extreme readings often precede mean reversion. Filters: 1d EMA trend for directional bias,
volume spike for confirmation. Target: 20-40 trades/year to minimize fee drag.
Works in both bull and bear markets by fading extremes in ranging conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h Williams %R(14)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    highest_high = np.full_like(high_4h, np.nan)
    lowest_low = np.full_like(low_4h, np.nan)
    
    for i in range(13, len(high_4h)):
        highest_high[i] = np.max(high_4h[i-13:i+1])
        lowest_low[i] = np.min(low_4h[i-13:i+1])
    
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        -100 * ((highest_high - close_4h) / (highest_high - lowest_low)),
        -50
    )
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 4h volume / 20-period average
    vol_ma_20 = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = df_4h['volume'].values / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        williams_val = williams_r_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        
        # Volume spike filter
        vol_threshold = 2.0
        
        if position == 0:
            # Enter long: oversold + above 1d EMA + volume spike
            if (williams_val < -80 and 
                ema_trend > 0 and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: overbought + below 1d EMA + volume spike
            elif (williams_val > -20 and 
                  ema_trend < 0 and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Williams %R returns to neutral range (-50 to -50) or opposite extreme
            if position == 1 and (williams_val > -50 or williams_val > -20):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (williams_val < -50 or williams_val < -80):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA34_Volume_Spike"
timeframe = "4h"
leverage = 1.0