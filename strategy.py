#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d HMA(21) trend filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for HMA trend and Donchian channels.
- Donchian channels calculated from previous 1d high/low (20-day lookback).
- Entry: Long when price breaks above upper Donchian with volume spike and close > 1d HMA21 (uptrend).
         Short when price breaks below lower Donchian with volume spike and close < 1d HMA21 (downtrend).
- Exit: When price returns to the opposite Donchian level (mean reversion) or ATR-based stoploss.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA calculation
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    close_series = pd.Series(close)
    wma_half = wma(close_series.values, half_period)
    wma_full = wma(close_series.values, period)
    
    # Handle array lengths
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    # Pad to original length
    result = np.full_like(close, np.nan)
    start_idx = len(close) - len(hma)
    if start_idx >= 0:
        result[start_idx:] = hma
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HMA trend and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1d HMA21 for trend filter
    hma_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    upper_donchian = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_donchian = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 4h
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    upper_donchian_aligned = align_htf_to_ltf(prices, df_1d, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_1d, lower_donchian)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(21, 20, 20)  # Need enough 1d bars for HMA21 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(hma_21_aligned[i]) or np.isnan(upper_donchian_aligned[i]) or 
            np.isnan(lower_donchian_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for breakout signals with volume spike and trend filter
            if volume_spike[i]:
                # Bullish breakout: price > upper Donchian and close > HMA21
                if close[i] > upper_donchian_aligned[i] and close[i] > hma_21_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakdown: price < lower Donchian and close < HMA21
                elif close[i] < lower_donchian_aligned[i] and close[i] < hma_21_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price returns to lower Donchian (mean reversion)
            if close[i] <= lower_donchian_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to upper Donchian (mean reversion)
            if close[i] >= upper_donchian_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dHMA21_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0