#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian breakout with 1d volume spike and chop regime filter.
    # Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period MA AND chop > 61.8 (range regime).
    # Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period MA AND chop > 61.8.
    # Exit when price returns to Donchian midpoint.
    # Uses discrete position sizing (0.25) to target 50-150 trades over 4 years.
    # Works in bull/bear via chop filter avoiding trend-following false signals in ranging markets.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume 20-period MA
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate True Range for chop calculation
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate 1d chop regime: ATR(14) / (highest high - lowest low over 14) * 100 * log10(sqrt(14))/log10(10)
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop = np.where(range_14 > 0, 100 * np.log10(atr_14 * np.sqrt(14) / range_14) / np.log10(10), 50)
    
    # Align 1d indicators to 12h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 12h Donchian channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_spike = volume_1d_aligned[i] > 1.5 * vol_ma_1d_aligned[i]
        
        # Chop regime filter: only trade in ranging markets (chop > 61.8)
        chop_filter = chop_aligned[i] > 61.8
        
        # Donchian breakout conditions
        breakout_long = close[i] > highest_high_20[i-1]  # Break above previous period high
        breakout_short = close[i] < lowest_low_20[i-1]   # Break below previous period low
        
        # Exit conditions: price returns to Donchian midpoint
        exit_long = close[i] < donchian_mid[i]
        exit_short = close[i] > donchian_mid[i]
        
        # Entry conditions
        if breakout_long and volume_spike and chop_filter and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_short and volume_spike and chop_filter and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "12h_1d_donchian_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0