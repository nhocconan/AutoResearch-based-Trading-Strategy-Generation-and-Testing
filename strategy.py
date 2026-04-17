#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d ATR-based volatility filter + Donchian(20) breakout + volume confirmation.
Long when price breaks above 20-day high with 1d ATR(14) > 1.5x 50-period ATR(14) MA and volume > 1.5x 20-day volume average.
Short when price breaks below 20-day low with same volatility and volume filters.
Uses 1d ATR to ensure breakouts occur during high volatility regimes, reducing false breakouts in choppy markets.
Designed to work in bull markets (breakouts with volatility expansion) and bear markets (breakdowns with volatility expansion).
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
    
    # Get 1d data for ATR and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14)
    def calculate_atr(high_vals, low_vals, close_vals, window):
        tr1 = high_vals[1:] - low_vals[1:]
        tr2 = np.abs(high_vals[1:] - close_vals[:-1])
        tr3 = np.abs(low_vals[1:] - close_vals[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        return atr
    
    atr_14_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1d ATR(14) 50-period moving average
    atr_ma_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 1d Donchian(20) channels
    def donchian_channel(high_vals, low_vals, window):
        upper = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high_1d, low_1d, 20)
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (4h)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_ma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50_1d)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for ATR and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(atr_ma_50_1d_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current 1d ATR > 1.5x 50-period ATR MA
        volatility_expansion = atr_14_1d_aligned[i] > 1.5 * atr_ma_50_1d_aligned[i]
        
        # Volume confirmation: current 4h volume > 1.5x 20-day volume average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above 20-day high with volatility expansion and volume
            if (close[i] > donchian_upper_aligned[i] and 
                volatility_expansion and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low with volatility expansion and volume
            elif (close[i] < donchian_lower_aligned[i] and 
                  volatility_expansion and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 20-day low (opposite side of channel)
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 20-day high (opposite side of channel)
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dATR_VolatilityExpansion_Donchian20_Breakout_Volume_Confirm"
timeframe = "4h"
leverage = 1.0