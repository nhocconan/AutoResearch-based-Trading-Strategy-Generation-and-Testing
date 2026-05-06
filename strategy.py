#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12-hour timeframe with 1-day Elder Ray (bull/bear power) and 200-period EMA trend filter
# Long when Bull Power > 0 and close > EMA200 (bullish momentum + uptrend)
# Short when Bear Power < 0 and close < EMA200 (bearish momentum + downtrend)
# Uses daily high/low for power calculations and 1-day EMA200 for trend filter
# Designed to capture momentum in both bull and bear markets by aligning with institutional sentiment
# Target: 50-150 total trades over 4 years = 12-37/year with 0.25 position sizing

name = "6h_1dElderRay_EMA200_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1-day data for Elder Ray and EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1-day EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA200, Bear Power = Low - EMA200
    bull_power_1d = df_1d['high'].values - ema200_1d
    bear_power_1d = df_1d['low'].values - ema200_1d
    
    # Align 1-day indicators to 6h timeframe
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any critical value is NaN
        if (np.isnan(ema200_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Bull Power > 0 (bullish momentum) AND price above EMA200 (uptrend)
            if bull_power_aligned[i] > 0 and close[i] > ema200_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power < 0 (bearish momentum) AND price below EMA200 (downtrend)
            elif bear_power_aligned[i] < 0 and close[i] < ema200_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power < 0 (momentum shifts bearish) OR price below EMA200 (trend break)
            if bear_power_aligned[i] < 0 or close[i] < ema200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power > 0 (momentum shifts bullish) OR price above EMA200 (trend break)
            if bull_power_aligned[i] > 0 or close[i] > ema200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals