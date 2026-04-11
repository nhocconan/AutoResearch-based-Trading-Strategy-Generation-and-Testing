#!/usr/bin/env python3
# 4h_1d_donchian_breakout_v1
# Strategy: 4h breakout of Donchian(20) channel calculated from 1d high/low, with volume confirmation and ADX trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Donchian channels based on daily high/low capture significant support/resistance levels.
# Breakouts above the upper channel or below the lower channel with above-average volume and trending market (ADX>25)
# capture momentum. Works in both bull and bear markets by following breakout direction. Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Donchian channels from 1d high/low (20-day period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper channel = highest high of last 20 days
    upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower channel = lowest low of last 20 days
    lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h (using previous 1d bar's values)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    # Volume confirmation: 1d volume > average (20-day)
    vol_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # ADX trend filter (14-period)
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original index
    
    # Calculate Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(adx[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 1d volume (aligned)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, vol_1d)[i]
        vol_confirm = vol_1d_current > vol_avg_20_aligned[i]
        
        # ADX trend filter: only trade in trending markets
        trend_filter = adx[i] > 25
        
        # Breakout conditions using Donchian channels
        breakout_above = close[i] > upper_aligned[i-1]  # break above upper channel
        breakout_below = close[i] < lower_aligned[i-1]  # break below lower channel
        
        long_signal = breakout_above and vol_confirm and trend_filter
        short_signal = breakout_below and vol_confirm and trend_filter
        
        # Exit conditions: opposite breakout or volume failure or trend weakness
        long_exit = close[i] < lower_aligned[i-1] or not vol_confirm or adx[i] < 20
        short_exit = close[i] > upper_aligned[i-1] or not vol_confirm or adx[i] < 20
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals