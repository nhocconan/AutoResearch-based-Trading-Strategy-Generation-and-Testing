#!/usr/bin/env python3
# 6h_1d_Donchian20_WeeklyPivot_Direction_Volume
# Hypothesis: 6h breakout above/below Donchian(20) with weekly pivot direction filter and volume confirmation.
# Long when price > Donchian(20) high and weekly pivot trend is up; short when price < Donchian(20) low and weekly pivot trend is down.
# Uses volume > 1.5x 20-period average for confirmation. Designed for low trade frequency (~20-50/year) to avoid fee drag.
# Weekly pivot trend derived from prior week's close: up if current week's close > prior week's close, down otherwise.
# Works in bull/bear via weekly pivot direction filter which adapts to longer-term trend.

name = "6h_1d_Donchian20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly pivot direction: 1 if current week close > prior week close, -1 otherwise
    close_1w = df_1w['close'].values
    weekly_trend = np.where(close_1w[1:] > close_1w[:-1], 1, -1)
    # Prepend first value (no prior week) as neutral (0) to avoid look-ahead
    weekly_trend = np.concatenate([[0], weekly_trend])
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Donchian(20) on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(weekly_trend_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above Donchian high with up weekly trend and volume surge
            if close[i] > highest_high[i] and weekly_trend_aligned[i] == 1 and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low with down weekly trend and volume surge
            elif close[i] < lowest_low[i] and weekly_trend_aligned[i] == -1 and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long when price drops below Donchian low
                if close[i] < lowest_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short when price rises above Donchian high
                if close[i] > highest_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals