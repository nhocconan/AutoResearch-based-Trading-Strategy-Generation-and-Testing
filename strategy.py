#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian20_Breakout_1wTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA100 trend filter
    close_1w = df_1w['close'].values
    ema100_1w = pd.Series(close_1w).ewm(span=100, adjust=False, min_periods=100).mean().values
    trend_1w = (close_1w > ema100_1w).astype(float)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Get 12h data once for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Donchian(20) channels on 12h timeframe
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: highest high of last 20 bars
    upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 bars
    lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (no additional delay needed)
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    
    # Volume filter: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above upper band with volume filter and 1w uptrend
            long_cond = (close[i] > upper_aligned[i] and vol_filter[i] and trend_1w_aligned[i] > 0.5)
            
            # Short entry: price breaks below lower band with volume filter and 1w downtrend
            short_cond = (close[i] < lower_aligned[i] and vol_filter[i] and trend_1w_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below lower band (reversal signal)
            if close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper band (reversal signal)
            if close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Donchian(20) breakout strategy with volume filter and 1w EMA100 trend filter.
# Uses 1w trend filter for multi-timeframe alignment to avoid counter-trend trades.
# Donchian breakouts capture strong momentum moves in both bull and bear markets.
# Volume filter (2x 20-period average) reduces false breakouts.
# Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.
# Works in bull markets (trend-following breakouts) and bear markets (mean reversion from extremes).