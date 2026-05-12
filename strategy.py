#!/usr/bin/env python3
# 1D_FIBONACCI_RETRACEMENT_1WTREND_VOLUME_CONFIRMATION
# Hypothesis: Fibonacci retracement levels (38.2%, 61.8%) from weekly swing points
# combined with weekly trend filter (EMA34) and volume spike confirmation.
# Works in bull/bear: weekly EMA ensures trend alignment, Fibonacci levels provide
# high-probability retracement zones, volume confirms institutional participation.
# Target: 10-25 trades/year.

name = "1D_FIBONACCI_RETRACEMENT_1WTREND_VOLUME_CONFIRMATION"
timeframe = "1d"
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
    
    # Weekly data for swing points and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly swing high and low (using 5-period window)
    swing_high = pd.Series(df_1w['high']).rolling(window=5, center=True, min_periods=5).max().values
    swing_low = pd.Series(df_1w['low']).rolling(window=5, center=True, min_periods=5).min().values
    
    # Fibonacci retracement levels: 38.2% and 61.8%
    fib_range = swing_high - swing_low
    fib_382 = swing_high - fib_range * 0.382
    fib_618 = swing_high - fib_range * 0.618
    
    # Weekly EMA for trend filter (34-period)
    ema34 = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Fibonacci levels and EMA to daily timeframe
    fib_382_aligned = align_htf_to_ltf(prices, df_1w, fib_382)
    fib_618_aligned = align_htf_to_ltf(prices, df_1w, fib_618)
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34)
    
    # Volume spike detection (20-period volume MA on daily)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > vol_ma * 2.0  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(fib_382_aligned[i]) or np.isnan(fib_618_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price retraces to 61.8% level with volume confirmation in uptrend
            if (close[i] <= fib_618_aligned[i] * 1.005 and  # Allow small buffer
                close[i] >= fib_618_aligned[i] * 0.995 and
                vol_spike[i] and 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price retraces to 38.2% level with volume confirmation in downtrend
            elif (close[i] <= fib_382_aligned[i] * 1.005 and
                  close[i] >= fib_382_aligned[i] * 0.995 and
                  vol_spike[i] and 
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches 38.2% level (take profit) or breaks below 61.8% (stop)
            if close[i] <= fib_382_aligned[i] * 1.005:  # Take profit at 38.2%
                signals[i] = 0.0
                position = 0
            elif close[i] < fib_618_aligned[i] * 0.995:  # Stop loss below 61.8%
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches 61.8% level (take profit) or breaks above 38.2% (stop)
            if close[i] >= fib_618_aligned[i] * 0.995:  # Take profit at 61.8%
                signals[i] = 0.0
                position = 0
            elif close[i] > fib_382_aligned[i] * 1.005:  # Stop loss above 38.2%
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals