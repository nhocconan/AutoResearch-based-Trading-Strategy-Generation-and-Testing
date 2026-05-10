#!/usr/bin/env python3
# 12h_Donchian_Breakout_1dTrend_With_Volume_Confirmation
# Hypothesis: Buy breakouts above 20-period Donchian high on 12h when 1d trend is up (close > EMA50),
# sell breakdowns below 20-period Donchian low on 12h when 1d trend is down (close < EMA50).
# Volume confirmation: current volume > 1.5x 20-period average volume.
# Designed for low-frequency, high-conviction trades to minimize fee drag.
# Targets ~20-30 trades/year per symbol.

name = "12h_Donchian_Breakout_1dTrend_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align 1d trend to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Donchian channels (20-period) on 12h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * avg_volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high in uptrend with volume confirmation
            if (high[i] > highest_high[i] and
                trend_1d_up_aligned[i] > 0.5 and
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low in downtrend with volume confirmation
            elif (low[i] < lowest_low[i] and
                  trend_1d_down_aligned[i] > 0.5 and
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: breakdown below Donchian low or trend reversal
            if (low[i] < lowest_low[i] or
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: breakout above Donchian high or trend reversal
            if (high[i] > highest_high[i] or
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals