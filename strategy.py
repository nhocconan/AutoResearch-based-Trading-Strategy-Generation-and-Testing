#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1d ADX trend filter.
# Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period average AND 1d ADX > 25 (trending market).
# Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period average AND 1d ADX > 25.
# Exit when price crosses back inside the Donchian channel.
# Designed to capture strong trends with volume confirmation, avoiding false breakouts in ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency to avoid fee drag.

name = "4h_Donchian_20_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for volume and ADX filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 30:
        return np.zeros(n)
    
    # Daily volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(df_d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_filter = df_d['volume'].values > (1.5 * vol_ma20)
    volume_filter_aligned = align_htf_to_ltf(prices, df_d, volume_filter)
    
    # Daily ADX calculation (14-period)
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # True Range
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_d[0] - low_d[0]  # First TR
    
    # Directional Movement
    up_move = high_d - np.roll(high_d, 1)
    down_move = np.roll(low_d, 1) - low_d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_filter = adx > 25
    adx_filter_aligned = align_htf_to_ltf(prices, df_d, adx_filter)
    
    # Donchian channel (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Sufficient warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_filter_aligned[i]) or np.isnan(adx_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, volume filter, ADX > 25
            long_cond = (close[i] > highest_high[i]) and volume_filter_aligned[i] and adx_filter_aligned[i]
            # Short conditions: price breaks below Donchian low, volume filter, ADX > 25
            short_cond = (close[i] < lowest_low[i]) and volume_filter_aligned[i] and adx_filter_aligned[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian low
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian high
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals