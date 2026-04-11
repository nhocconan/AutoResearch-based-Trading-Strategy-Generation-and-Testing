#!/usr/bin/env python3
# 12h_1d_keltner_channel_v1
# Strategy: 12h Keltner Channel breakout with 1w trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Price breaking above/below Keltner Channel (EMA-based) during strong weekly trend
# with volume confirmation captures breakouts in both bull and bear markets. Weekly trend filter
# ensures we only trade in the direction of higher timeframe momentum, reducing whipsaw.
# Volume confirmation adds conviction to breakouts. Designed for low frequency (15-25 trades/year)
# to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_keltner_channel_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily data for ATR (used in Keltner Channel)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ATR(14) for Keltner Channel width
    tr1 = high_1d[1:] - low_1d[:-1]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.insert(tr, 0, 0)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # EMA(20) for Keltner Channel middle line (using daily data)
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bounds
    upper_keltner = ema_20 + 2 * atr
    lower_keltner = ema_20 - 2 * atr
    
    # Align daily Keltner levels to 12h timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    
    # Volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_keltner_aligned[i]) or 
            np.isnan(lower_keltner_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation (current volume > 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Entry conditions
        if weekly_uptrend and close[i] > upper_keltner_aligned[i] and volume_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        elif weekly_downtrend and close[i] < lower_keltner_aligned[i] and volume_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite weekly trend or price returns to middle
        elif position == 1 and (not weekly_uptrend or close[i] < ema_20[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not weekly_downtrend or close[i] > ema_20[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals