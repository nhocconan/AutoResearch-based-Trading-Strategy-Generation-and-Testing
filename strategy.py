#!/usr/bin/env python3
# 6h_1D_Fibonacci_Extension_Trend_Follow
# Hypothesis: Uses Fibonacci extension (127.2%, 161.8%) from 1-day swing high/low to identify momentum continuation zones.
# In bull markets, price tends to extend beyond swing highs; in bear markets, extends below swing lows.
# Trend filter (1-day EMA34) ensures trades align with higher timeframe direction.
# Volume confirmation filters low-momentum breakouts.
# Targets 12-30 trades per year by requiring confluence of Fibonacci extension, trend, and volume.

name = "6h_1D_Fibonacci_Extension_Trend_Follow"
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
    
    # Volume spike: >1.8x 20-period average (avoid noise)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Daily data for swing points and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter (needs only completed daily bar)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily swing high/low (prior completed day)
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    range_1d = prev_high_1d - prev_low_1d
    
    # Fibonacci extension levels: 127.2% and 161.8% of prior day's range
    fib_ext_127 = prev_close_1d + 1.272 * range_1d  # upward extension
    fib_ext_161 = prev_close_1d + 1.618 * range_1d  # stronger upward
    fib_ext_neg_127 = prev_close_1d - 1.272 * range_1d  # downward extension
    fib_ext_neg_161 = prev_close_1d - 1.618 * range_1d  # stronger downward
    
    # Align all daily values to 6h timeframe (only after daily bar closes)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    fib_ext_127_aligned = align_htf_to_ltf(prices, df_1d, fib_ext_127)
    fib_ext_161_aligned = align_htf_to_ltf(prices, df_1d, fib_ext_161)
    fib_ext_neg_127_aligned = align_htf_to_ltf(prices, df_1d, fib_ext_neg_127)
    fib_ext_neg_161_aligned = align_htf_to_ltf(prices, df_1d, fib_ext_neg_161)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is not yet available
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(fib_ext_127_aligned[i]) or 
            np.isnan(fib_ext_161_aligned[i]) or
            np.isnan(fib_ext_neg_127_aligned[i]) or
            np.isnan(fib_ext_neg_161_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 127.2% extension + volume spike + price above daily EMA34 (uptrend)
            if (close[i] > fib_ext_127_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below -127.2% extension + volume spike + price below daily EMA34 (downtrend)
            elif (close[i] < fib_ext_neg_127_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below 127.2% extension OR closes below daily EMA34
            if close[i] < fib_ext_127_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above -127.2% extension OR closes above daily EMA34
            if close[i] > fib_ext_neg_127_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals