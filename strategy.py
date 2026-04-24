#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation.
- Primary timeframe: 6h for execution, HTF: 12h for EMA50 trend.
- Donchian breakout: long when price > 20-bar high, short when price < 20-bar low.
- Trend filter: 12h EMA50 slope positive for longs, negative for shorts.
- Volume confirmation: current volume > 1.5x 20-period volume MA.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via breakouts in uptrend, in bear via breakdowns in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian(20) on 6h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20) + 1  # Need EMA50(50)+buffer, Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trend filter: EMA50 slope
            if not np.isnan(ema_50_12h_aligned[i-1]):
                ema_slope = ema_50_12h_aligned[i] - ema_50_12h_aligned[i-1]
                # Long: uptrend + breakout + volume
                if ema_slope > 0 and close[i] > highest_high[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: downtrend + breakdown + volume
                elif ema_slope < 0 and close[i] < lowest_low[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price retests 20-bar low or trend reversal
            if close[i] < lowest_low[i] or (not np.isnan(ema_50_12h_aligned[i-1]) and ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price retests 20-bar high or trend reversal
            if close[i] > highest_high[i] or (not np.isnan(ema_50_12h_aligned[i-1]) and ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0