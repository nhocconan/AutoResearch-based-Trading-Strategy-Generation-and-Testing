#!/usr/bin/env python3
# 12h_1d_adx_breakout_v1
# Strategy: 12h Donchian(20) breakout with ADX trend filter and 1d volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture momentum. ADX > 25 ensures trending markets to avoid whipsaws.
# 1d volume > 1.5x 20-period average confirms institutional participation. Designed for low frequency
# (15-25 trades/year) to minimize fee drag in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_adx_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # 1d volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # ADX calculation (14-period)
    # True Range
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(adx[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        trending = adx[i] > 25
        
        # Volume confirmation: 1d volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Entry logic: Donchian breakout + trend + volume
        if (close[i] > highest_high[i] and  # Break above upper band
            trending and vol_confirm and position != 1):
            position = 1
            signals[i] = 0.25
        elif (close[i] < lowest_low[i] and  # Break below lower band
              trending and vol_confirm and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: opposite Donchian breakout or loss of trend/volume
        elif position == 1 and (close[i] < lowest_low[i] or not trending or not vol_confirm):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > highest_high[i] or not trending or not vol_confirm):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals