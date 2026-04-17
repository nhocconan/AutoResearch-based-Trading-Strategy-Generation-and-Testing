#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1d volume spike and chop regime filter.
Long when price breaks above Donchian upper (20) AND 1d volume > 1.5x 20-bar average AND chop > 61.8 (range regime).
Short when price breaks below Donchian lower (20) AND 1d volume > 1.5x 20-bar average AND chop > 61.8.
Exit when price touches Donchian midpoint or opposite band.
Uses 1d for volume and chop filters, 4h for Donchian calculation and execution.
Designed to capture mean reversion in ranging markets with volume confirmation. Target: 20-40 trades/year per symbol.
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
    
    # Get 1d data for volume and chop regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume MA for confirmation
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d chop regime (choppiness index)
    # CHOP = 100 * log10(sum(ATR(14))) / log10(14)
    tr1 = np.maximum(high_1d - low_1d, 
                     np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                np.abs(low_1d - np.roll(close_1d, 1))))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    log_sum = np.log10(sum_atr14 + 1e-10)
    log_n = np.log10(14)
    chop = 100 * log_sum / log_n
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Align all 1d indicators to 4h timeframe
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-bar average
        volume_confirmed = volume_1d[i // 16] > 1.5 * vol_ma_20_1d_aligned[i] if i // 16 < len(volume_1d) else False
        
        # Chop regime filter: only trade in ranging markets (chop > 61.8)
        chop_filter = chop_aligned[i] > 61.8
        
        # Breakout conditions
        breakout_upper = close[i] > highest_high[i]
        breakout_lower = close[i] < lowest_low[i]
        
        # Exit conditions: touch midpoint or opposite band
        touch_mid = abs(close[i] - donchian_mid[i]) < 0.001 * close[i]  # within 0.1%
        touch_opposite = (position == 1 and close[i] < lowest_low[i]) or \
                         (position == -1 and close[i] > highest_high[i])
        
        if position == 0:
            # Long: break above upper band with volume confirmation and chop regime
            if (breakout_upper and volume_confirmed and chop_filter):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume confirmation and chop regime
            elif (breakout_lower and volume_confirmed and chop_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch midpoint or break below lower band
            if (touch_mid or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch midpoint or break above upper band
            if (touch_mid or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_Chop_Regime"
timeframe = "4h"
leverage = 1.0