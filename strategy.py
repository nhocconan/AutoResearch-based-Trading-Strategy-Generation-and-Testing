#!/usr/bin/env python3
"""
4h_Donchian_Breakout_VolumeTrend_Filter
Hypothesis: Capture breakouts from Donchian channels with volume confirmation and 1d trend filter.
Works in bull markets via long breakouts above upper band, in bear markets via short breakouts below lower band.
Uses 1d EMA50 for trend direction to avoid counter-trend trades. Volume filter ensures breakout strength.
Target: 20-30 trades/year to minimize fee drag.
"""

name = "4h_Donchian_Breakout_VolumeTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Donchian Channel (20-period) ===
    lookback = 20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Rolling max/min with lookback period (not centered)
    donchian_high = high_series.rolling(window=lookback, min_periods=lookback).max().shift(1).values
    donchian_low = low_series.rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # === 1d EMA50 Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Volume Spike Filter ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5  # 1.5x average volume
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers Donchian and EMA calculation)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_4h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above Donchian high + above 1d EMA50 + volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema50_1d_4h[i] and volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: Break below Donchian low + below 1d EMA50 + volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema50_1d_4h[i] and volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Reverse signal or stop loss via opposite Donchian band
            if position == 1:
                if close[i] < donchian_low[i]:  # Reverse signal
                    signals[i] = -position_size
                    position = -1
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > donchian_high[i]:  # Reverse signal
                    signals[i] = position_size
                    position = 1
                else:
                    signals[i] = -position_size
    
    return signals