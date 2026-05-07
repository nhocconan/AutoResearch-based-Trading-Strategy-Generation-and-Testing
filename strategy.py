#!/usr/bin/env python3
# 4h_Donchian20_Volume_Trend_Signal_v2
# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation.
# Long when price breaks above upper Donchian band, price > 1d EMA50, and volume > 1.5x average.
# Short when price breaks below lower Donchian band, price < 1d EMA50, and volume > 1.5x average.
# Exit on opposite band touch. Designed for 4h timeframe to target 20-50 trades/year.
# Works in bull markets via breakouts and in bear via short breakdowns with trend filter.

name = "4h_Donchian20_Volume_Trend_Signal_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike detection: 1.5x average volume (20-period for stability)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure we have Donchian and EMA50 data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian, price > 1d EMA50 (uptrend), volume spike
            if (close[i] > upper[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, price < 1d EMA50 (downtrend), volume spike
            elif (close[i] < lower[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to or below lower Donchian (opposite band)
            if close[i] <= lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to or above upper Donchian (opposite band)
            if close[i] >= upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals