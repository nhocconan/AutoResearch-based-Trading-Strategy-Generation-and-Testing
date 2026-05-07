#!/usr/bin/env python3
"""
6h_Keltner_Channel_Breakout_1wTrend_v1
Hypothesis: On 6h timeframe, use Keltner Channel breakouts filtered by weekly trend direction.
Long when price breaks above upper KC and weekly trend is bullish.
Short when price breaks below lower KC and weekly trend is bearish.
Keltner Channels adapt to volatility, reducing false breakouts in low-volatility periods.
Works in both bull and bear markets by requiring alignment with weekly trend.
"""
name = "6h_Keltner_Channel_Breakout_1wTrend_v1"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Keltner Channel parameters
    kc_period = 20
    atr_period = 10
    kc_multiplier = 2.0
    
    # Calculate ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate EMA for middle line
    ema_middle = pd.Series(close).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    
    # Calculate upper and lower bands
    kc_upper = ema_middle + (kc_multiplier * atr)
    kc_lower = ema_middle - (kc_multiplier * atr)
    
    # Weekly EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(kc_period, atr_period, 50)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper KC + weekly uptrend
            if close[i] > kc_upper[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower KC + weekly downtrend
            elif close[i] < kc_lower[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below middle line
            if close[i] < ema_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        else:  # position == -1
            # Exit short: price crosses above middle line
            if close[i] > ema_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals