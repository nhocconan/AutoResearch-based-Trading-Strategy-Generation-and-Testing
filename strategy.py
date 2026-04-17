#!/usr/bin/env python3
"""
4h_Donchian20_VolumeTrend_v1
Hypothesis: Breakout strategy using Donchian channel (20) on 4h, filtered by 1d EMA trend and volume confirmation. Long when price breaks above upper band in uptrend (close > EMA50), short when breaks below lower band in downtrend (close < EMA50). Volume must be above 1.5x 20-period average. Position size 0.25. Designed to capture trends in both bull and bear markets with low trade frequency to minimize fee drag.
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
    
    # === 1d data for trend filter and volume average ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d volume average (20-period)
    vol_avg20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg20_1d)
    
    # === 4h Donchian channel (20) ===
    # Upper band: highest high over last 20 periods
    high_series = pd.Series(high)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over last 20 periods
    low_series = pd.Series(low)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    
    # Warmup: covers EMA50, Donchian, and volume average
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_avg20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 1d volume
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_filter = vol_1d_current > 1.5 * vol_avg20_1d_aligned[i]
        
        # Entry conditions: Donchian breakout in trend direction
        if position == 0:
            # Long: break above upper band in uptrend (close > EMA50) with volume
            if close[i] > donchian_upper[i] and close[i] > ema50_1d_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below lower band in downtrend (close < EMA50) with volume
            elif close[i] < donchian_lower[i] and close[i] < ema50_1d_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse when price crosses opposite Donchian band
        elif position == 1:
            if close[i] < donchian_lower[i]:  # exit long when price breaks below lower band
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if close[i] > donchian_upper[i]:  # exit short when price breaks above upper band
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeTrend_v1"
timeframe = "4h"
leverage = 1.0