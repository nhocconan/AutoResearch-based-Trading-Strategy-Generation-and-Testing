#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeRegime
Hypothesis: 4h Donchian(20) breakout with 1d EMA trend filter and volume/regime confirmation. Uses HTF 1d for EMA trend alignment (price > 1d EMA34 for long, < 1d EMA34 for short) to reduce whipsaw. Volume confirmation requires >1.5x 20-bar mean volume. Choppiness regime filter: CHOP(14) < 61.8 for trending markets. Targets 20-30 trades/year per symbol by requiring strong volume spike, clear trend, and trending regime. Designed to work in both bull (breakouts with volume) and bear (trend-following shorts) markets via disciplined entry/exit.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-bar mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 1.5)
    
    # Choppiness regime filter: CHOP(14) < 61.8 for trending markets
    tr = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - pd.Series(close).shift(1)), np.abs(low - pd.Series(close).shift(1)))))
    atr_14 = tr.rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 / np.log10(14) / (highest_high_14 - lowest_low_14))
    chop[np.isnan(chop) | np.isinf(chop)] = 50  # neutral value for invalid
    trending_regime = chop < 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian, EMA, volume, and chop
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_mean_20[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high in uptrend (price > 1d EMA34) with volume confirmation and trending regime
            # Short: price breaks below Donchian low in downtrend (price < 1d EMA34) with volume confirmation and trending regime
            long_signal = (close[i] > high_20[i]) and (close[i] > ema_34_aligned[i]) and vol_confirm[i] and trending_regime[i]
            short_signal = (close[i] < low_20[i]) and (close[i] < ema_34_aligned[i]) and vol_confirm[i] and trending_regime[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below Donchian low (breakout failure)
            exit_signal = close[i] < low_20[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Donchian high (breakout failure)
            exit_signal = close[i] > high_20[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeRegime"
timeframe = "4h"
leverage = 1.0