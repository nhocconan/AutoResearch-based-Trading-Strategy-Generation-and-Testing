#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
Long when price breaks above 20-period 1d Donchian upper band AND 1d ATR(14) < 30-period 1d ATR mean (low volatility regime) AND 12h volume > 1.5x 20-bar average volume.
Short when price breaks below 20-period 1d Donchian lower band AND 1d ATR(14) < 30-period 1d ATR mean AND 12h volume > 1.5x 20-bar average volume.
Exit when price touches the 20-period 1d Donchian midpoint (mean of upper and lower bands).
Uses 1d for Donchian channels and ATR regime, 12h for execution and volume confirmation.
Designed to capture breakouts from low volatility contractions in any market regime. Target: 15-25 trades/year per symbol.
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
    
    # Get 1d data for Donchian channels and ATR regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian(20) channels
    lookback = 20
    upper_20 = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    lower_20 = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    mid_20 = (upper_20 + lower_20) / 2.0
    
    # Calculate 1d ATR(14) for volatility regime
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR mean for regime filter (30-period mean of ATR)
    atr_mean_30 = pd.Series(atr_14).rolling(window=30, min_periods=30).mean().values
    
    # Calculate 12h volume MA for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    mid_20_aligned = align_htf_to_ltf(prices, df_1d, mid_20)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_mean_30_aligned = align_htf_to_ltf(prices, df_1d, atr_mean_30)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or
            np.isnan(mid_20_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or
            np.isnan(atr_mean_30_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-bar average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Volatility regime: current ATR < ATR mean (low volatility)
        low_vol_regime = atr_14_aligned[i] < atr_mean_30_aligned[i]
        
        # Breakout conditions
        breakout_upper = close[i] > upper_20_aligned[i]
        breakout_lower = close[i] < lower_20_aligned[i]
        
        # Exit condition: touch midpoint
        touch_mid = abs(close[i] - mid_20_aligned[i]) < 0.001 * close[i]  # within 0.1%
        
        if position == 0:
            # Long: break above upper band with volume confirmation and low volatility regime
            if (breakout_upper and volume_confirmed and low_vol_regime):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume confirmation and low volatility regime
            elif (breakout_lower and volume_confirmed and low_vol_regime):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch midpoint
            if touch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch midpoint
            if touch_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Volume_ATRRegime"
timeframe = "12h"
leverage = 1.0