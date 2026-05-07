#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_Volume_V1
Hypothesis: Use 12h Donchian(20) breakouts with 1d EMA50 trend filter and volume > 1.5x average.
Target 15-25 trades/year by requiring strong trend alignment and volume confirmation.
Designed to work in both bull (breakouts with trend) and bear (fades from extremes with trend) markets.
Works on BTC/ETH/SOL via strong trend + volume confirmation.
"""
name = "12h_Donchian20_Breakout_1dTrend_Volume_V1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and Donchian
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5 * 100-period average
    vol_avg = pd.Series(volume).rolling(window=100, min_periods=100).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for Donchian and EMA50
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + 1d uptrend + volume filter
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + 1d downtrend + volume filter
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite Donchian level
            if position == 1:
                if close[i] <= donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals