#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with daily ADX trend filter and volume confirmation
# Uses Donchian(20) breakouts for entry, daily ADX(14) > 25 to filter trending markets,
# and volume > 1.5x 20-period average for confirmation. Exits on opposite Donchian(10) break.
# Designed for moderate trade frequency (target: 20-50 trades/year) with strong trend capture.
# Works in bull markets via long breakouts and bear markets via short breakdowns.

name = "4h_donchian20_daily_adx_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period for entry, 10-period for exit)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    highest_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Calculate ADX(14) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_14 = adx  # ADX(14)
    
    # Align daily ADX to 4h
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(highest_10[i]) or np.isnan(lowest_10[i]) or
            np.isnan(adx_14_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_14_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Long entry: price breaks above Donchian(20) high
        long_entry = close[i] > highest_20[i]
        
        # Short entry: price breaks below Donchian(20) low
        short_entry = close[i] < lowest_20[i]
        
        # Long exit: price breaks below Donchian(10) low
        long_exit = close[i] < lowest_10[i]
        
        # Short exit: price breaks above Donchian(10) high
        short_exit = close[i] > highest_10[i]
        
        # Generate signals with trend and volume filters
        if trending and vol_confirmed:
            if long_entry:
                signals[i] = 0.30  # Long 30%
            elif short_entry:
                signals[i] = -0.30  # Short 30%
            elif long_exit and signals[i-1] > 0:
                signals[i] = 0.0  # Exit long
            elif short_exit and signals[i-1] < 0:
                signals[i] = 0.0  # Exit short
            else:
                signals[i] = signals[i-1]  # Hold position
        else:
            # No trend or no volume confirmation: flatten
            signals[i] = 0.0
    
    return signals