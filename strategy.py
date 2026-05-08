#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day Fibonacci retracement levels with volume confirmation and trend filter.
# Fibonacci levels (38.2%, 61.8%) act as strong support/resistance in trending markets.
# Long when price pulls back to 61.8% level in uptrend with volume confirmation.
# Short when price bounces from 38.2% level in downtrend with volume confirmation.
# Uses 1-week trend filter to ensure alignment with higher timeframe momentum.
# Designed for low trade frequency (10-20/year) to minimize fade whipsaw and capture high-probability pullbacks.

name = "6h_FibPullback_TrendFilter_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Fibonacci level calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Fibonacci retracement levels from prior day's swing
    fib_618 = np.zeros_like(close_1d)  # 61.8% retracement (support in uptrend)
    fib_382 = np.zeros_like(close_1d)  # 38.2% retracement (resistance in downtrend)
    
    for i in range(1, len(close_1d)):
        # Prior day's high and low
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        
        # Range
        rng = ph - pl
        
        # Fibonacci levels (using close as anchor for consistency)
        fib_618[i] = pl + (rng * 0.618)  # 61.8% level
        fib_382[i] = pl + (rng * 0.382)  # 38.2% level
    
    # First day has no prior data
    fib_618[0] = fib_382[0] = np.nan
    
    # Align Fibonacci levels to 6h timeframe
    fib_618_aligned = align_htf_to_ltf(prices, df_1d, fib_618)
    fib_382_aligned = align_htf_to_ltf(prices, df_1d, fib_382)
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA(21) for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_trend_up = ema_21_1w[1:] > ema_21_1w[:-1]  # Rising weekly EMA
    weekly_trend_up = np.concatenate([[False], weekly_trend_up])  # Align with daily index
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    
    # 6x EMA(50) for intermediate trend and dynamic support/resistance
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.8x 30-period EMA
    vol_ema = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    vol_confirm = volume > (vol_ema * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA(50)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(fib_618_aligned[i]) or np.isnan(fib_382_aligned[i]) or
            np.isnan(ema_50[i]) or np.isnan(weekly_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: pullback to 61.8% Fib level in uptrend with volume
            if (weekly_trend_aligned[i] > 0.5 and  # Weekly uptrend
                close[i] > ema_50[i] and             # Above intermediate EMA
                close[i] <= fib_618_aligned[i] * 1.005 and  # Near 61.8% level (allow 0.5% slack)
                close[i] >= fib_618_aligned[i] * 0.995 and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: bounce from 38.2% Fib level in downtrend with volume
            elif (weekly_trend_aligned[i] <= 0.5 and  # Weekly downtrend
                  close[i] < ema_50[i] and            # Below intermediate EMA
                  close[i] >= fib_382_aligned[i] * 0.995 and  # Near 38.2% level
                  close[i] <= fib_382_aligned[i] * 1.005 and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below 38.2% level or trend turns down
            if close[i] < fib_382_aligned[i] * 0.995 or weekly_trend_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above 61.8% level or trend turns up
            if close[i] > fib_618_aligned[i] * 1.005 or weekly_trend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals