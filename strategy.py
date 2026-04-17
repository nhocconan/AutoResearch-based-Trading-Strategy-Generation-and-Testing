#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d volume spike and ADX trend filter.
Long when price breaks above 20-bar 12h high AND 1d volume > 1.5x 20-bar average AND ADX(14) > 25.
Short when price breaks below 20-bar 12h low AND 1d volume > 1.5x 20-bar average AND ADX(14) > 25.
Exit when price touches 12h midpoint of the channel or opposite breakout level.
Uses 1d for volume confirmation and ADX regime filter, 12h for execution and Donchian channels.
Designed to capture strong trending moves with volume confirmation in both bull and bear markets.
Target: 12-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation and ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h Donchian channels (20-bar)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    midpoint = (highest_high + lowest_low) / 2
    
    # Calculate 1d volume MA for confirmation
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX(14) for trend strength
    # True Range
    tr1 = np.maximum(high_1d - low_1d, 
                     np.absolute(high_1d - np.roll(close_1d, 1)),
                     np.absolute(low_1d - np.roll(close_1d, 1)))
    tr1[0] = high_1d[0] - low_1d[0]
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/14)
    tr14 = pd.Series(tr1).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    # DX and ADX
    dx = 100 * np.absolute(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align all 1d indicators to 12h timeframe
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(midpoint[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-bar average
        volume_confirmed = volume_1d[i // 12] > 1.5 * vol_ma_20_1d_aligned[i] if i // 12 < len(volume_1d) else False
        
        # Regime filter: ADX > 25 indicates trending market
        trending_market = adx_aligned[i] > 25
        
        # Breakout conditions
        breakout_high = close[i] > highest_high[i]
        breakout_low = close[i] < lowest_low[i]
        
        # Exit conditions: touch midpoint or opposite level
        touch_midpoint = abs(close[i] - midpoint[i]) < 0.001 * close[i]  # within 0.1%
        touch_opposite = (position == 1 and close[i] < lowest_low[i]) or \
                         (position == -1 and close[i] > highest_high[i])
        
        if position == 0:
            # Long: break above highest high with volume confirmation and trending market
            if (breakout_high and volume_confirmed and trending_market):
                signals[i] = 0.25
                position = 1
            # Short: break below lowest low with volume confirmation and trending market
            elif (breakout_low and volume_confirmed and trending_market):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch midpoint or break below lowest low
            if (touch_midpoint or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch midpoint or break above highest high
            if (touch_midpoint or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dVolume_ADX_Trend"
timeframe = "12h"
leverage = 1.0