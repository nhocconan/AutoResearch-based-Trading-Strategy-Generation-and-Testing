#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Donchian(20) breakout + volume confirmation + ATR-based stoploss.
Long when price breaks above 20-period 1d Donchian high with volume > 1.5x 20-period 1d volume average.
Short when price breaks below 20-period 1d Donchian low with volume > 1.5x 20-period 1d volume average.
Uses discrete position sizing 0.25 to limit fee drag. Target: 50-150 total trades over 4 years.
Donchian channels provide structural breakout levels; volume confirms institutional participation.
Designed to work in bull markets (breakout continuation) and bear markets (mean reversion after volatility expansion).
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
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 12h
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # need enough for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above 1d Donchian high with volume confirmation
            if (close[i] > donchian_high_20_aligned[i] and volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian low with volume confirmation
            elif (close[i] < donchian_low_20_aligned[i] and volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 1d Donchian low (mean reversion)
            if close[i] < donchian_low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 1d Donchian high (mean reversion)
            if close[i] > donchian_high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dDonchian20_Volume_Confirm_MeanReversion"
timeframe = "12h"
leverage = 1.0