#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1d ADX trend filter.
# Long when price breaks above Donchian high(20) AND 1d volume > 1.5x 20-day average AND 1d ADX > 25 (trending market).
# Short when price breaks below Donchian low(20) AND 1d volume > 1.5x 20-day average AND 1d ADX > 25.
# Exit when price crosses back inside the Donchian channel.
# Uses 4h timeframe with 1d volume and ADX for higher timeframe context.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency to avoid fee drag.

name = "4h_Donchian_20_1dVolume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h Donchian channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily data for volume and ADX
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 30:
        return np.zeros(n)
    
    # Daily volume filter: current volume > 1.5x 20-day average
    vol_ma20 = pd.Series(df_d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_filter_raw = df_d['volume'].values > (1.5 * vol_ma20)
    volume_filter = align_htf_to_ltf(prices, df_d, volume_filter_raw)
    
    # Daily ADX (14-period) for trend strength
    # Calculate True Range
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_d[0] - low_d[0]  # First TR
    
    # Directional Movement
    up_move = high_d - np.roll(high_d, 1)
    down_move = np.roll(low_d, 1) - low_d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_filter_raw = adx > 25
    adx_filter = align_htf_to_ltf(prices, df_d, adx_filter_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(adx_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, volume filter, ADX > 25
            long_cond = (close[i] > donch_high[i]) and volume_filter[i] and adx_filter[i]
            # Short conditions: price breaks below Donchian low, volume filter, ADX > 25
            short_cond = (close[i] < donch_low[i]) and volume_filter[i] and adx_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian low
            if close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Donchian high
            if close[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals