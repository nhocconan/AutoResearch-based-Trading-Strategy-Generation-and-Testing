#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w ADX trend filter and volume confirmation
# Long when price breaks above 20-day high AND volume > 1.5x 20-day average AND 1w ADX > 25 (strong trend)
# Short when price breaks below 20-day low AND volume > 1.5x 20-day average AND 1w ADX > 25 (strong trend)
# Exit when price crosses back to 10-day midpoint OR 1w ADX < 20 (weakening trend)
# Uses discrete sizing (0.30) to limit fee drag. Target: 15-25 trades/year per symbol.
# Donchian channels provide clear trend structure, volume confirms conviction, 1w ADX filters for trending markets
# to avoid whipsaws in ranging conditions. Works in bull markets via longs and bear markets via shorts.

name = "1d_Donchian20_VolumeSpike_1wADX_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian channels calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 1d data (using 20-period lookback)
    # We use rolling window on close prices to avoid look-ahead
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (already aligned, but keep for consistency)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate 10-day midpoint for exit
    midpoint_10 = (high_20_aligned + low_20_aligned) / 2
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate ADX on 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = pd.Series(high_1w - low_1w)
    tr2 = pd.Series(abs(high_1w - pd.Series(close_1w).shift(1)))
    tr3 = pd.Series(abs(low_1w - pd.Series(close_1w).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1w - pd.Series(high_1w).shift(1))
    down_move = pd.Series(pd.Series(low_1w).shift(1) - low_1w)
    up_move = up_move.where((up_move > down_move) & (up_move > 0), 0)
    down_move = down_move.where((down_move > up_move) & (down_move > 0), 0)
    
    # Directional Indicators
    plus_di = 100 * pd.Series(up_move).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(down_move).rolling(window=14, min_periods=14).mean().values / atr
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1w ADX to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: volume > 1.5x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above 20-day high AND volume spike AND 1w ADX > 25
            if (close[i] > high_20_aligned[i] and 
                volume_filter[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below 20-day low AND volume spike AND 1w ADX > 25
            elif (close[i] < low_20_aligned[i] and 
                  volume_filter[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses back to 10-day midpoint OR 1w ADX < 20 (weakening trend)
            if (close[i] < midpoint_10[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses back to 10-day midpoint OR 1w ADX < 20 (weakening trend)
            if (close[i] > midpoint_10[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals