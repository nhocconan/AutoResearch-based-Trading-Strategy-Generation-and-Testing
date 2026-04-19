#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with daily volume confirmation and ADX trend filter.
# Long when price breaks above 20-period high with daily volume > 1.5x average and ADX > 25 (trending).
# Short when price breaks below 20-period low with daily volume > 1.5x average and ADX > 25.
# Exit on opposite Donchian break or ADX < 20 (range).
# Uses daily volume and ADX to filter breakouts, ensuring institutional participation in trending markets.
# Target: 20-50 trades/year per symbol (~80-200 total over 4 years).
name = "4h_Donchian20_1dVol_ADX_Trend"
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
    
    # Get daily data for volume and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on 4h data
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume average (20-period)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX on daily data (14-period)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])
    down_move = np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align daily indicators to 4h timeframe (wait for daily close)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20, 14+14)  # Donchian, volume MA, ADX smoothing
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_roll_max[i]) or np.isnan(low_roll_min[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_channel = high_roll_max[i]
        lower_channel = low_roll_min[i]
        vol_ma = vol_ma_aligned[i]
        adx_val = adx_aligned[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Trend filter: ADX > 25 for trending market
        trending = adx_val > 25
        
        if position == 0:
            # Enter long: price breaks above upper channel AND volume confirmed AND trending
            if price > upper_channel and volume_confirmed and trending:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower channel AND volume confirmed AND trending
            elif price < lower_channel and volume_confirmed and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below lower channel OR ADX < 20 (range)
            if price < lower_channel or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above upper channel OR ADX < 20 (range)
            if price > upper_channel or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals