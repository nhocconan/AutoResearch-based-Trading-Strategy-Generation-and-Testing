#!/usr/bin/env python3
# 12h_1w_donchian_breakout_volume_v1
# Strategy: 12h Donchian Channel breakout with volume confirmation and 1w ADX trend filter
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture momentum. Volume confirms breakout strength. 1w ADX > 25 ensures trending markets. Works in bull (breakouts up) and bear (breakouts down). Low trade frequency (~15-30/year) minimizes fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_donchian_breakout_volume_v1"
timeframe = "12h"
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
    
    # Load 1w data ONCE before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1w ADX(14) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    adx_1w = adx  # Already aligned to 1w
    
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # 12h Donchian Channel (20)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: 1w ADX > 25 indicates trending market
        trending = adx_1w_aligned[i] > 25
        
        # Entry logic: Donchian breakout + volume + trend filter
        if close[i] > donch_high[i] and vol_confirm[i] and trending and position != 1:
            position = 1
            signals[i] = 0.25
        elif close[i] < donch_low[i] and vol_confirm[i] and trending and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite Donchian breach (trailing stop)
        elif position == 1 and close[i] < donch_low[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > donch_high[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals