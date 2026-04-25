#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dATRTrend_VolumeConfirm_v1
Hypothesis: Trade Donchian(20) breakouts on 4h with 1d ATR-based trend filter and volume confirmation. In bullish 1d trend (close > close[-20] + 0.5*ATR(14)), buy when price breaks above upper Donchian; in bearish 1d trend (close < close[-20] - 0.5*ATR(14)), sell when price breaks below lower Donchian. Volume spike (1.5x 20-bar avg) confirms participation. Uses discrete position sizing (0.25) to minimize fee drag and target ~20-40 trades/year. Designed to work in both bull and bear markets by following the higher timeframe trend.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d trend: bullish if close > close[-20] + 0.5*ATR(14)
    close_shift_20 = np.concatenate([[np.nan]*20, close_1d[:-20]])
    atr_shift_20 = np.concatenate([[np.nan]*20, atr_14[:-20]])
    bullish_threshold = close_shift_20 + 0.5 * atr_shift_20
    bearish_threshold = close_shift_20 - 0.5 * atr_shift_20
    trend_bullish = close_1d > bullish_threshold
    trend_bearish = close_1d < bearish_threshold
    
    # Align 1d trend to 4h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish.astype(float))
    
    # Calculate Donchian(20) on 4h
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 1.5x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trend_bullish_aligned[i]) or 
            np.isnan(trend_bearish_aligned[i]) or
            np.isnan(high_ma[i]) or
            np.isnan(low_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend
        tb = trend_bullish_aligned[i] > 0.5
        tbd = trend_bearish_aligned[i] > 0.5
        
        if position == 0:
            # Look for breakout signals with volume confirmation and trend alignment
            long_signal = close[i] > high_ma[i] and volume_spike[i] and tb
            short_signal = close[i] < low_ma[i] and volume_spike[i] and tbd
            
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
            # Exit when price breaks below lower Donchian or trend reverses
            exit_signal = close[i] < low_ma[i] or not tb
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price breaks above upper Donchian or trend reverses
            exit_signal = close[i] > high_ma[i] or not tbd
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dATRTrend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0