#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wTrend_Volume
Hypothesis: Uses 20-period Donchian channel breakouts on 12h timeframe, filtered by 1-week trend direction and volume confirmation.
Works in bull markets by capturing breakouts in uptrends and in bear markets by shorting breakdowns in downtrends.
Target: 15-30 trades/year per symbol.
"""

name = "12h_Donchian20_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # Convert to Series for indicator calculations
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # Donchian Channel (20-period)
    highest_high = high_s.rolling(window=20, min_periods=20).max()
    lowest_low = low_s.rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1w trend filter: EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w_up = close_1w > ema34_1w
    trend_1w_down = close_1w < ema34_1w
    
    # Align 1w trend to 12h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        # Conditions for long entry: break above upper band + uptrend + volume
        if position == 0:
            if (close[i] > donchian_upper[i] and
                trend_1w_up_aligned[i] > 0.5 and
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Conditions for short entry: break below lower band + downtrend + volume
            elif (close[i] < donchian_lower[i] and
                  trend_1w_down_aligned[i] > 0.5 and
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below midpoint OR trend changes
            midpoint = (donchian_upper[i] + donchian_lower[i]) / 2
            if (close[i] < midpoint or trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above midpoint OR trend changes
            midpoint = (donchian_upper[i] + donchian_lower[i]) / 2
            if (close[i] > midpoint or trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals