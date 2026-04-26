#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeSpike_v1
Hypothesis: Camarilla pivot (R1/S1) breakout with 4h EMA50 trend filter and volume confirmation.
Only long when price breaks above R1 and close > 4h EMA50, short when price breaks below S1 and close < 4h EMA50.
Uses discrete position sizing (0.0, ±0.20) to minimize fee churn. Target: 60-150 total trades over 4 years (15-37/year).
Designed to work in both bull and bear markets by combining pivot breakouts with trend and volume filters.
"""

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
    
    # Calculate Camarilla pivots for previous day
    # Typical price for previous day (using rolling window)
    typical_price = (high + low + close) / 3.0
    # For intraday, we use previous day's OHLC - approximate with rolling 26-period (1 day of 1h bars)
    lookback = 26  # ~1 day of 1h bars
    if n < lookback:
        return np.zeros(n)
    
    # Calculate rolling high, low, close for previous day
    # We need to shift to avoid look-ahead: use previous completed day
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # Previous day's OHLC (using rolling window and shift)
    prev_high = high_series.rolling(window=lookback, min_periods=lookback).max().shift(lookback).values
    prev_low = low_series.rolling(window=lookback, min_periods=lookback).min().shift(lookback).values
    prev_close = close_series.rolling(window=lookback, min_periods=lookback).mean().shift(lookback).values
    
    # Camarilla levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Load 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(lookback, 50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Discrete position sizing
        base_size = 0.20
        
        # Long logic: price breaks above R1 + close > 4h EMA50 + volume spike
        if close[i] > r1[i] and close[i] > ema_50_4h_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S1 + close < 4h EMA50 + volume spike
        elif close[i] < s1[i] and close[i] < ema_50_4h_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: price returns to pivot or loss of volume confirmation
        elif position == 1 and (close[i] < pivot[i] or not volume_spike[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > pivot[i] or not volume_spike[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0