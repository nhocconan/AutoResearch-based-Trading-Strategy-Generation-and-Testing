#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d Donchian breakout + volume confirmation + ATR stop.
Long when price breaks above 1d Donchian(20) high with volume > 1.5x 20-period average and ATR(14) > 0.
Short when price breaks below 1d Donchian(20) low with same conditions.
Uses discrete position sizing 0.25 to limit fee drag. Target: 75-200 total trades over 4 years.
Donchian channels provide structural support/resistance; volume confirms participation; ATR ensures sufficient volatility.
Designed to work in bull markets (breakout continuation) and bear markets (strong trend continuation).
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
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) for 1d
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all to 4h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.5 * vol_ma_20_1d_aligned[i]
        # Volatility filter: ATR > 0 ensures sufficient volatility
        sufficient_volatility = atr_aligned[i] > 0
        
        if position == 0:
            # Long: price breaks above 1d Donchian high with volume and volatility
            if (close[i] > donchian_high_aligned[i] and 
                volume_confirmed and 
                sufficient_volatility):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian low with volume and volatility
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_confirmed and 
                  sufficient_volatility):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 1d Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 1d Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dDonchian20_Volume_Confirm_ATRFilter"
timeframe = "4h"
leverage = 1.0