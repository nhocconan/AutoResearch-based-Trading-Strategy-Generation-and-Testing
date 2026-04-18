#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d ADX trend filter + volume confirmation
# Donchian channel (20) identifies breakouts in both directions
# 1d ADX > 25 ensures we only trade in trending markets (works in bull/bear)
# Volume > 1.5x 20-period average confirms breakout conviction
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag
# Works in bull markets (upward breakouts with trend) and bear markets (downward breakouts with trend)
name = "4h_Donchian20_1dADX25_VolumeConfirm"
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
    
    # Get 1d data for ADX filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX components on 1d data
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d.shift(1))
    tr3 = np.abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    plus_dm = high_1d.diff()
    minus_dm = low_1d.diff().abs()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    # Smooth TR and DM (14-period)
    tr_ma = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_ma = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_ma = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_ma / tr_ma
    minus_di = 100 * minus_dm_ma / tr_ma
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND trending AND volume confirmation
            if close[i] > highest_high[i] and trending and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND trending AND volume confirmation
            elif close[i] < lowest_low[i] and trending and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian lower band OR trend weakens
            if close[i] < lowest_low[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian upper band OR trend weakens
            if close[i] > highest_high[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals