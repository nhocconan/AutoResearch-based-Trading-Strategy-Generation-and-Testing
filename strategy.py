#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 1d volatility filter and ADX trend strength.
In bull markets: breakouts capture momentum. In bear markets: ADX filter avoids false signals during low-volatility periods.
Uses 1d ATR ratio (current ATR / 20-day ATR) to filter for expanding volatility environments.
Only takes longs when price breaks above Donchian high (20) in expanding vol + strong trend (ADX>25).
Only takes shorts when price breaks below Donchian low (20) in expanding vol + strong trend (ADX>25).
Target: 20-50 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for volatility filter and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR (14-period)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], 
                           np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-day ATR average for volatility ratio
    atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    # Volatility expansion: current ATR > 1.2 * 20-day average ATR
    vol_expansion = atr_1d > (atr_ma_20 * 1.2)
    vol_expansion_aligned = align_htf_to_ltf(prices, df_1d, vol_expansion.astype(float))
    
    # Calculate 1d ADX (14-period) for trend strength
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # DI values
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    # Strong trend: ADX > 25
    strong_trend = adx > 25
    strong_trend_aligned = align_htf_to_ltf(prices, df_1d, strong_trend.astype(float))
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_expansion_aligned[i]) or 
            np.isnan(strong_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Donchian breakout + volatility expansion + strong trend
        breakout_long = close[i] > donchian_high[i]
        breakout_short = close[i] < donchian_low[i]
        vol_filter = vol_expansion_aligned[i] > 0.5
        trend_filter = strong_trend_aligned[i] > 0.5
        
        long_entry = breakout_long and vol_filter and trend_filter
        short_entry = breakout_short and vol_filter and trend_filter
        
        # Exit when price returns to midpoint of Donchian channel
        donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
        exit_long = position == 1 and close[i] < donchian_mid
        exit_short = position == -1 and close[i] > donchian_mid
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_vol_adx"
timeframe = "4h"
leverage = 1.0