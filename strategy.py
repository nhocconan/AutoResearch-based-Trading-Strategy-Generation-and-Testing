#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 1-day volume confirmation and 1-week ADX filter.
# Donchian breakouts capture breakout momentum in trending markets.
# Volume confirmation ensures institutional participation.
# 1-week ADX > 25 filters for strong trending conditions only, avoiding choppy markets.
# Works in bull markets (upward breakouts) and bear markets (downward breakouts).
# Target: 20-50 trades/year (80-200 total over 4 years) to stay within optimal range.
name = "4h_Donchian20_1dVolume_1wADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 20-period average volume on 1d
    vol_1d = pd.Series(df_1d['volume'].values)
    vol_ma_20_1d = vol_1d.rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ADX on 1w data
    high_1w = pd.Series(df_1w['high'].values)
    low_1w = pd.Series(df_1w['low'].values)
    close_1w = pd.Series(df_1w['close'].values)
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = abs(high_1w - close_1w.shift(1))
    tr3 = abs(low_1w - close_1w.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_1w.diff()
    down_move = low_1w.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_1w)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_1w)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1w = dx.ewm(alpha=1/14, adjust=False).mean().values
    
    # Calculate Donchian channels on 4h data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align 1d volume MA to 4h timeframe
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Align 1w ADX to 4h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5 * 20-day average volume
        volume_confirm = volume[i] > (1.5 * vol_ma_20_1d_aligned[i])
        
        # Strong trend filter: ADX > 25
        strong_trend = adx_1w_aligned[i] > 25
        
        if position == 0:
            # Long: Price breaks above upper Donchian band + volume confirmation + strong trend
            if close[i] > high_max_20[i] and volume_confirm and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian band + volume confirmation + strong trend
            elif close[i] < low_min_20[i] and volume_confirm and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below lower Donchian band OR ADX weakens
            if close[i] < low_min_20[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above upper Donchian band OR ADX weakens
            if close[i] > high_max_20[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals