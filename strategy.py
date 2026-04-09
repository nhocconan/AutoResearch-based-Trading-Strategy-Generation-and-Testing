#!/usr/bin/env python3
# 6h_weekly_pivot_donchian_volume_v3
# Hypothesis: 6h strategy using weekly pivot points for trend direction and 6h Donchian(20) breakouts with volume confirmation.
# In trending markets, price continues in pivot direction after breaking Donchian channels.
# Volume confirmation filters false breakouts. Weekly pivot provides higher-timeframe bias.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 12-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_donchian_volume_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    # Weekly support/resistance levels
    r1 = 2 * weekly_pivot - low_1w
    s1 = 2 * weekly_pivot - high_1w
    r2 = weekly_pivot + (high_1w - low_1w)
    s2 = weekly_pivot - (high_1w - low_1w)
    r3 = high_1w + 2 * (weekly_pivot - low_1w)
    s3 = low_1w - 2 * (high_1w - weekly_pivot)
    
    # Determine weekly bias: price above/below weekly pivot
    weekly_bias = np.where(close_1w > weekly_pivot, 1, -1)  # 1=bullish, -1=bearish
    
    # Align weekly data to 6h timeframe
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r2 + (high_1w - low_1w))  # R4 = R2 + range
    s4_aligned = align_htf_to_ltf(prices, df_1w, s2 - (high_1w - low_1w))  # S4 = S2 - range
    
    # 6h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(weekly_bias_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price moves below Donchian low or weekly bias turns bearish
            if close[i] < low_20[i] or weekly_bias_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above Donchian high or weekly bias turns bullish
            if close[i] > high_20[i] or weekly_bias_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed:
                # Long entry: price breaks above Donchian high with volume AND weekly bullish bias
                if close[i] > high_20[i] and weekly_bias_aligned[i] == 1:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian low with volume AND weekly bearish bias
                elif close[i] < low_20[i] and weekly_bias_aligned[i] == -1:
                    position = -1
                    signals[i] = -0.25
    
    return signals