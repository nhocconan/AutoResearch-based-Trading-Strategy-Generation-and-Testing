#!/usr/bin/env python3
name = "6h_ElderRay_BullBear_1dTrend_WeakFilter"
timeframe = "6h"
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
    
    # 1d data for Elder Ray and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 13-period EMA for Elder Ray (1d)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    # 13-period EMA for trend filter (1d)
    ema13_trend = ema13_1d.copy()
    
    # Bull Power = High - EMA13
    bull_power = high_1d - ema13_1d
    # Bear Power = Low - EMA13
    bear_power = low_1d - ema13_1d
    
    # Align to 6h timeframe (use same-day values, available after 1d close)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema13_trend_aligned = align_htf_to_ltf(prices, df_1d, ema13_trend)
    
    # Weak filter: require Bull Power > 0 and Bear Power < 0 for strong signal
    # Actually, we want clarity: Bull Power rising AND Bear Power falling
    # But simpler: require Bull Power > 0 for long, Bear Power < 0 for short
    # And trend filter: price > EMA13 for long, < EMA13 for short
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 13  # for EMA warmup
    
    for i in range(start_idx, n):
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(ema13_trend_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (bulls in control) + price above EMA13 (uptrend)
            if bull_power_aligned[i] > 0 and close[i] > ema13_trend_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bears in control) + price below EMA13 (downtrend)
            elif bear_power_aligned[i] < 0 and close[i] < ema13_trend_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power turns negative OR price breaks below EMA13
            if bull_power_aligned[i] <= 0 or close[i] < ema13_trend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns positive OR price breaks above EMA13
            if bear_power_aligned[i] >= 0 or close[i] > ema13_trend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals