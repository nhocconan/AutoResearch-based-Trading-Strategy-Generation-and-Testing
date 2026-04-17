#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with daily Donchian channel breakout + volume confirmation + ATR regime filter.
Long when price breaks above 20-period Donchian upper band with volume > 1.3x 20-period average and ATR(14) < ATR(50) (low volatility regime).
Short when price breaks below 20-period Donchian lower band with volume > 1.3x 20-period average and ATR(14) < ATR(50).
Exit when price touches the opposite Donchian band or ATR(14) > ATR(50) (high volatility regime).
Uses discrete sizing 0.25 to minimize fee churn. Designed to work in both bull and bear markets by filtering for low volatility breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_ma_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).mean().values
    low_ma_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).mean().values
    donchian_upper = high_ma_20
    donchian_lower = low_ma_20
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    tr2 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(tr2, np.abs(low_1d[1:] - close_1d[:-1]))
    tr2 = np.concatenate([[np.nan], tr2])
    atr50 = pd.Series(tr2).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 1d volume 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 12h
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    atr50_aligned = align_htf_to_ltf(prices, df_1d, atr50)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for ATR50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(atr14_aligned[i]) or np.isnan(atr50_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.3 * vol_ma_20_aligned[i]
        
        # Regime filter: ATR(14) < ATR(50) (low volatility regime)
        low_vol_regime = atr14_aligned[i] < atr50_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper band with volume and low vol regime
            if (close[i] > donchian_upper_aligned[i] and 
                volume_confirmed and 
                low_vol_regime):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band with volume and low vol regime
            elif (close[i] < donchian_lower_aligned[i] and 
                  volume_confirmed and 
                  low_vol_regime):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches Donchian lower band or high volatility regime
            if (close[i] < donchian_lower_aligned[i] or 
                not low_vol_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches Donchian upper band or high volatility regime
            if (close[i] > donchian_upper_aligned[i] or 
                not low_vol_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dDonchian20_Volume_ATRRegime"
timeframe = "12h"
leverage = 1.0