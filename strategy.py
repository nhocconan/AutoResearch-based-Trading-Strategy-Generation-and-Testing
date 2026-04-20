#!/usr/bin/env python3
# 4h_1d_Donchian20_Breakout_Volume_Trend
# Hypothesis: On 4h timeframe, trade Donchian(20) breakouts with volume confirmation and daily EMA trend filter.
# Designed to work in both bull and bear markets by aligning trades with daily trend.
# Targets 20-50 trades per year to avoid overtrading and fee drag.

name = "4h_1d_Donchian20_Breakout_Volume_Trend"
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
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'])
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high, volume spike, and price above daily EMA34 (uptrend)
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * volume_ma[i] and
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, volume spike, and price below daily EMA34 (downtrend)
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * volume_ma[i] and
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend reversal (below EMA34)
            if close[i] < donchian_low[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend reversal (above EMA34)
            if close[i] > donchian_high[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals