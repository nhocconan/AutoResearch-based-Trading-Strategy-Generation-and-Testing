#!/usr/bin/env python3
# 1h_Momentum_With_4hTrend_Volume
# Hypothesis: On 1h timeframe, enter long when price breaks above 4h high with volume, short when breaks below 4h low with volume. Use 4h trend (EMA50) to filter direction. This captures momentum in trending markets while avoiding counter-trend trades. Designed for low frequency (~20-40 trades/year) to minimize fee drag.

name = "1h_Momentum_With_4hTrend_Volume"
timeframe = "1h"
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
    
    # 4h data for trend and range
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA50 for trend
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_4h_up = close_4h > ema50_4h
    trend_4h_down = close_4h < ema50_4h
    
    # Align 4h trend to 1h
    trend_4h_up_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up.astype(float))
    trend_4h_down_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_down.astype(float))
    
    # 4h high and low for breakout levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_4h_aligned = align_htf_to_ltf(prices, df_4h, high_4h)
    low_4h_aligned = align_htf_to_ltf(prices, df_4h, low_4h)
    
    # Volume confirmation: 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trend_4h_up_aligned[i]) or np.isnan(trend_4h_down_aligned[i]) or
            np.isnan(high_4h_aligned[i]) or np.isnan(low_4h_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: price breaks above 4h high with 4h uptrend and volume
            if (close[i] > high_4h_aligned[i] and 
                trend_4h_up_aligned[i] > 0.5 and volume_confirm):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below 4h low with 4h downtrend and volume
            elif (close[i] < low_4h_aligned[i] and 
                  trend_4h_down_aligned[i] > 0.5 and volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit when price returns to 4h midpoint or trend fails
            mid_4h = (high_4h_aligned[i] + low_4h_aligned[i]) / 2
            if (close[i] < mid_4h or 
                trend_4h_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit when price returns to 4h midpoint or trend fails
            mid_4h = (high_4h_aligned[i] + low_4h_aligned[i]) / 2
            if (close[i] > mid_4h or 
                trend_4h_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals