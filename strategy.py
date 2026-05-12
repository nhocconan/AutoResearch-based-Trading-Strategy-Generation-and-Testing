#!/usr/bin/env python3
name = "6h_ElderRay_BullBearPower_1dTrendFilter"
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
    
    # Load daily data once for trend filter and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA(13) for trend filter
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Daily high/low for Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align Elder Ray components to 6h (wait for daily close)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema13_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (strong buying pressure) + daily uptrend
            if bull_power_aligned[i] > 0 and close[i] > ema13_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (strong selling pressure) + daily downtrend
            elif bear_power_aligned[i] < 0 and close[i] < ema13_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power turns negative (buying pressure fading)
            if bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns positive (selling pressure fading)
            if bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals