#!/usr/bin/env python3
"""
6h_LongTerm_Momentum_12hTrend_Volume
Hypothesis: 6-hour momentum (6-month high breakout) aligned with 12-hour EMA50 trend and volume confirmation captures breakouts with follow-through in trending markets. Works in bull (breakouts above 12h EMA50) and bear (breakdowns below 12h EMA50). Low-frequency signals via 6-month lookback to minimize noise and overtrading.
"""
name = "6h_LongTerm_Momentum_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6-month high (approx 180 trading days at 6h: 180*4 = 720 bars)
    period_6m = 720
    high_6m = pd.Series(high).rolling(window=period_6m, min_periods=period_6m).max().values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period_6m, 50)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(high_6m[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 6m high + 12h uptrend + volume
            if close[i] > high_6m[i] and close[i] > ema_50_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 6m low + 12h downtrend + volume
            # Calculate 6m low similarly
            low_6m = pd.Series(low).rolling(window=period_6m, min_periods=period_6m).min().values
            if close[i] < low_6m[i] and close[i] < ema_50_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses back through 6m high/low in opposite direction
            if position == 1:
                if close[i] < high_6m[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                low_6m = pd.Series(low).rolling(window=period_6m, min_periods=period_6m).min().values
                if close[i] > low_6m[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals