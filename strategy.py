#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Volume_ATRTrend
Hypothesis: Donchian(20) breakouts on 4h with 1d ATR-based trend filter and volume confirmation (>1.5x 20-bar avg).
Trades long when price breaks above upper band in uptrend (ATR(14) rising), short when breaks below lower band in downtrend (ATR(14) falling).
Uses discrete position sizing (0.25) to minimize fee churn. Designed for 20-50 trades/year to work in both bull and bear markets via trend alignment and volatility expansion.
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
    
    # Get 1d data for HTF trend filter (ATR-based)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate ATR(14) on 1d for trend filter
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR trend: rising ATR = increasing volatility (trending market), falling ATR = decreasing volatility (choppy)
    atr_rising = atr_1d > np.roll(atr_1d, 1)
    atr_falling = atr_1d < np.roll(atr_1d, 1)
    # Handle first value
    atr_rising[0] = False
    atr_falling[0] = False
    
    # Align HTF ATR trend to 4h timeframe
    atr_rising_aligned = align_htf_to_ltf(prices, df_1d, atr_rising, additional_delay_bars=1)
    atr_falling_aligned = align_htf_to_ltf(prices, df_1d, atr_falling, additional_delay_bars=1)
    
    # Donchian(20) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: 1.5x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20), volume MA (20), ATR (14)
    start_idx = max(lookback, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(atr_rising_aligned[i]) or
            np.isnan(atr_falling_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals with ATR trend filter and volume spike
            # Long: price breaks above upper band in rising volatility (trending up) with volume spike
            # Short: price breaks below lower band in falling volatility (trending down) with volume spike
            long_signal = (close[i] > highest_high[i]) and atr_rising_aligned[i] and volume_spike[i]
            short_signal = (close[i] < lowest_low[i]) and atr_falling_aligned[i] and volume_spike[i]
            
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
            # Exit when price moves back below Donchian midpoint (take profit at midpoint)
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            exit_signal = close[i] < midpoint
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Donchian midpoint (take profit at midpoint)
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            exit_signal = close[i] > midpoint
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_Volume_ATRTrend"
timeframe = "4h"
leverage = 1.0