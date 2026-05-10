#!/usr/bin/env python3
"""
4H_InsideBar_Breakout_1dTrend_Volume
Hypothesis: Uses daily inside bar (IB) patterns on 4h timeframe for breakout entries,
confirmed by 1d EMA50 trend and volume spike. Inside bars indicate consolidation
and low volatility; breakouts from these ranges often precede strong moves.
Works in both bull and bear markets by following 1d trend direction. Uses
discrete position sizing (0.25) to minimize fee churn. Target: 20-40 trades/year.
"""

name = "4H_InsideBar_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1d data for inside bar detection (requires previous day's high/low)
    # Inside bar: current day's high <= previous day's high AND low >= previous day's low
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    inside_bar = (df_1d['high'] <= prev_high) & (df_1d['low'] >= prev_low)
    # Convert to 1 if inside bar, 0 otherwise
    inside_bar_signal = inside_bar.astype(int).values
    # Align to 4h timeframe (use previous day's inside bar signal)
    inside_bar_aligned = align_htf_to_ltf(prices, df_1d, inside_bar_signal)
    
    # Volume filter: volume > 2.0x 20-period average on 4h chart
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(inside_bar_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_1d_aligned[i]
        price_below_ema = close[i] < ema_1d_aligned[i]
        
        if position == 0:
            # Long entry: inside bar breakout up + above 1d EMA + volume spike
            if (inside_bar_aligned[i] == 1 and 
                close[i] > df_1d['high'].shift(1).values[i] and  # Break above prev day high
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: inside bar breakout down + below 1d EMA + volume spike
            elif (inside_bar_aligned[i] == 1 and 
                  close[i] < df_1d['low'].shift(1).values[i] and  # Break below prev day low
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below previous day's low or volume drops
            if (close[i] < df_1d['low'].shift(1).values[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above previous day's high or volume drops
            if (close[i] > df_1d['high'].shift(1).values[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals