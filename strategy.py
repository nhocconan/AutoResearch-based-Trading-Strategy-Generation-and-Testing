#!/usr/bin/env python3
# 12h_1d_donchian_volume_breakout_v1
# Strategy: 12h Donchian breakout with volume confirmation and 1d ADX trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture sustained moves; volume confirmation filters false breakouts; 1d ADX > 25 ensures trend strength. Works in bull (breakouts up) and bear (breakouts down). Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_donchian_volume_breakout_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and DM calculation
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((high_diff > -low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((-low_diff > high_diff) & (-low_diff > 0), -low_diff, 0)
    tr = np.maximum(np.abs(high_diff), np.abs(low_diff))
    tr = np.maximum(tr, np.abs(np.diff(close_1d, prepend=close_1d[0])))
    
    # Smoothed averages
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).sum() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).sum() / atr)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    adx_1d_values = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_values)
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Price channels for entry/exit (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(volume_ma.iloc[i]) or 
            np.isnan(high_20.iloc[i]) or np.isnan(low_20.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend strength filter
        strong_trend = adx_1d_aligned[i] > 25
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * volume_ma.iloc[i]
        
        # Entry conditions
        if strong_trend and vol_confirm and close[i] > high_20.iloc[i-1] and position != 1:
            position = 1
            signals[i] = 0.25
        elif strong_trend and vol_confirm and close[i] < low_20.iloc[i-1] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: trend weakening or opposite breakout
        elif position == 1 and (adx_1d_aligned[i] < 20 or close[i] < low_20.iloc[i-1]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (adx_1d_aligned[i] < 20 or close[i] > high_20.iloc[i-1]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals