#!/usr/bin/env python3
# Hypothesis: 4h timeframe with 1-day Keltner Channel breakout and 1-week EMA trend filter.
# In low volatility regimes (ATR-based Keltner width narrow), price tends to break out with momentum.
# Enters long when price closes above Keltner upper band and price > weekly EMA50.
# Enters short when price closes below Keltner lower band and price < weekly EMA50.
# Uses 1-week EMA50 as trend filter to avoid counter-trend trades in strong trends.
# Exits when price crosses back below/above the Keltner middle line (20 EMA).
# Target: 75-200 total trades over 4 years (19-50/year) with size 0.25.

name = "4h_Keltner_Breakout_WeeklyEMA_Trend"
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
    
    # Calculate 1-day Keltner Channel (20, 1.5)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    # EMA20 for middle line
    ema_20 = close_1d.ewm(span=20, adjust=False, min_periods=20).mean()
    # ATR for band width
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=20, adjust=False, min_periods=20).mean()
    
    kc_upper = ema_20 + 1.5 * atr
    kc_lower = ema_20 - 1.5 * atr
    kc_middle = ema_20
    
    kc_upper_values = kc_upper.values
    kc_lower_values = kc_lower.values
    kc_middle_values = kc_middle.values
    
    kc_upper_aligned = align_htf_to_ltf(prices, df_1d, kc_upper_values)
    kc_lower_aligned = align_htf_to_ltf(prices, df_1d, kc_lower_values)
    kc_middle_aligned = align_htf_to_ltf(prices, df_1d, kc_middle_values)
    
    # Price position relative to Keltner Channel
    price_above_upper = close > kc_upper_aligned
    price_below_lower = close < kc_lower_aligned
    price_above_middle = close > kc_middle_aligned
    price_below_middle = close < kc_middle_aligned
    
    # Calculate 1-week EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close']
    ema_50 = close_1w.ewm(span=50, adjust=False, min_periods=50).mean()
    ema_50_values = ema_50.values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kc_upper_aligned[i]) or np.isnan(kc_lower_aligned[i]) or
            np.isnan(kc_middle_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(price_above_upper[i]) or np.isnan(price_below_lower[i]) or
            np.isnan(price_above_middle[i]) or np.isnan(price_below_middle[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price closes above Keltner upper AND price > weekly EMA50 (uptrend)
            if price_above_upper[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price closes below Keltner lower AND price < weekly EMA50 (downtrend)
            elif price_below_lower[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Keltner middle
            if price_below_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Keltner middle
            if price_above_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals