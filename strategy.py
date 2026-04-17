#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Donchian channel breakout + volume confirmation + ADX trend filter.
Long when price breaks above 1d Donchian upper channel (20) with volume > 1.3x 20-period average and ADX(14) > 25.
Short when price breaks below 1d Donchian lower channel (20) with volume > 1.3x 20-period average and ADX(14) > 25.
Exit when price returns to 1d Donchian middle (mean of upper/lower) or ADX < 20 (trend weak).
Donchian channels capture volatility-based structure; breakouts with volume and trend filter reduce false signals in both bull and bear markets.
Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag. Uses discrete sizing 0.25.
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
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_len = 20
    upper_1d = pd.Series(high_1d).rolling(window=donchian_len, min_periods=donchian_len).max().values
    lower_1d = pd.Series(low_1d).rolling(window=donchian_len, min_periods=donchian_len).min().values
    middle_1d = (upper_1d + lower_1d) / 2.0
    
    # Calculate 1d ADX (14-period)
    adx_len = 14
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift(1)).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=adx_len, adjust=False, min_periods=adx_len).mean()
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean() / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 4h
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    middle_1d_aligned = align_htf_to_ltf(prices, df_1d, middle_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(200, 50)  # need enough for ADX and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or 
            np.isnan(middle_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.3 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper channel with volume and strong trend (ADX > 25)
            if (close[i] > upper_1d_aligned[i] and 
                volume_confirmed and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower channel with volume and strong trend (ADX > 25)
            elif (close[i] < lower_1d_aligned[i] and 
                  volume_confirmed and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle channel or trend weakens (ADX < 20)
            if (close[i] < middle_1d_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle channel or trend weakens (ADX < 20)
            if (close[i] > middle_1d_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dDonchian20_Volume_ADX"
timeframe = "4h"
leverage = 1.0