#!/usr/bin/env python3
# 4h_Donchian_Breakout_Volume_Trend
# Hypothesis: Buy breakouts above 20-bar Donchian high in uptrends with volume confirmation.
# Sell short breakdowns below 20-bar Donchian low in downtrends with volume confirmation.
# Uses 1-day EMA34 for trend filter to work in both bull and bear markets.
# Volume spike (>1.5x average) confirms breakout strength.
# Targets ~25 trades/year to minimize fee drag.

name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align 1d trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # ensures 20-period indicators are valid
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high in uptrend with volume spike
            if (high[i] > highest_high[i] and
                trend_1d_up_aligned[i] > 0.5 and
                volume[i] > 1.5 * vol_avg[i]):
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low in downtrend with volume spike
            elif (low[i] < lowest_low[i] and
                  trend_1d_down_aligned[i] > 0.5 and
                  volume[i] > 1.5 * vol_avg[i]):
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