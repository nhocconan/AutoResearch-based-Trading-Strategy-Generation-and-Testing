#!/usr/bin/env python3
"""
6h_Weekly_High_Low_Pullback_Trend
Hypothesis: In trending markets, price pulls back to weekly highs/lows before continuing the trend.
Buy near weekly low in uptrend, sell near weekly high in downtrend. Uses weekly structure for support/resistance
and 6 EMA for trend filter. Works in bull (buy pullbacks) and bear (sell rallies) markets.
Target: 20-40 trades/year with strict pullback conditions to avoid overtrading.
"""

name = "6h_Weekly_High_Low_Pullback_Trend"
timeframe = "6h"
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
    
    # Get weekly data for high/low
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly high and low
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Align weekly high/low to 6h chart (wait for weekly close)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Trend filter: 6-period EMA on close
    ema6 = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Pullback definition: price within 0.5% of weekly level
    pullback_threshold = 0.005
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Pullback to weekly low in uptrend (price above EMA6)
            if (close[i] >= weekly_low_aligned[i] * (1 - pullback_threshold) and
                close[i] <= weekly_low_aligned[i] * (1 + pullback_threshold) and
                close[i] > ema6[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Pullback to weekly high in downtrend (price below EMA6)
            elif (close[i] <= weekly_high_aligned[i] * (1 + pullback_threshold) and
                  close[i] >= weekly_high_aligned[i] * (1 - pullback_threshold) and
                  close[i] < ema6[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal (price crosses below EMA6) or reached weekly high
            if (close[i] < ema6[i]) or (close[i] >= weekly_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal (price crosses above EMA6) or reached weekly low
            if (close[i] > ema6[i]) or (close[i] <= weekly_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals