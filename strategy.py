#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and ATR-based regime filter.
Long when price breaks above Donchian upper with volume > 1.3x 1d average volume AND ATR(14) < ATR(50) (low volatility regime).
Short when price breaks below Donchian lower with volume > 1.3x 1d average volume AND ATR(14) < ATR(50).
Exit when price touches the opposite Donchian level or ATR(14) > 1.5x ATR(50) (high volatility exit).
Uses 1d for volume and volatility regime confirmation, 12h for entry timing and Donchian calculation.
Designed to capture breakouts in low volatility environments which often precede strong moves in both bull and bear markets.
Target: 12-30 trades/year per symbol (50-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d volume MA(20)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # 1d ATR(14) and ATR(50) for regime filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr_14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    tr2 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(tr2, np.abs(low_1d[1:] - close_1d[:-1]))
    tr2 = np.concatenate([[np.nan], tr2])
    atr_50 = pd.Series(tr2).rolling(window=50, min_periods=50).mean().values
    
    atr_low_regime = atr_14 < atr_50  # Low volatility regime
    atr_low_regime[0] = False
    atr_low_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_low_regime)
    
    atr_high_exit = atr_14 > 1.5 * atr_50  # High volatility exit
    atr_high_exit[0] = False
    atr_high_exit_aligned = align_htf_to_ltf(prices, df_1d, atr_high_exit)
    
    # 12h Donchian(20) - we'll calculate this directly from 12h prices
    # Since we're on 12h timeframe, we can use the prices directly
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(lookback, 20, 50)  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(atr_low_regime_aligned[i]) or
            np.isnan(atr_high_exit_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 1d average volume
        volume_confirmed = volume[i] > 1.3 * vol_ma_20_aligned[i]
        
        # Donchian levels
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        
        # Breakout conditions
        breakout_upper = close[i] > upper_channel
        breakout_lower = close[i] < lower_channel
        
        if position == 0:
            # Long: break above upper channel with volume confirmation and low volatility regime
            if (breakout_upper and volume_confirmed and atr_low_regime_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel with volume confirmation and low volatility regime
            elif (breakout_lower and volume_confirmed and atr_low_regime_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches lower channel OR high volatility exit
            if (close[i] <= lower_channel) or atr_high_exit_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches upper channel OR high volatility exit
            if (close[i] >= upper_channel) or atr_high_exit_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dVolume_ATR_Regime"
timeframe = "12h"
leverage = 1.0