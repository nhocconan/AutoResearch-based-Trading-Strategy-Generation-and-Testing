#!/usr/bin/env python3
# Hypothesis: 6h timeframe strategy using 1d Fibonacci retracement levels (38.2% and 61.8%) from the previous day's range for mean reversion entries, with 1d trend filter (EMA50) to align with higher timeframe direction. Entries occur when price touches the Fibonacci level and shows rejection (close opposite to the level direction) with volume confirmation. This targets mean reversion in ranging markets while avoiding counter-trend trades in strong trends. Designed to work in both bull and bear markets by adapting to the 1d trend.

name = "6h_FibRetracement_382_618_1dEMA50_Trend_Volume"
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
    
    # Get 1d data for Fibonacci levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate previous day's range for Fibonacci levels
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    # Set first values to NaN
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Fibonacci retracement levels (38.2% and 61.8%)
    range_val = prev_high - prev_low
    fib_382 = prev_high - 0.382 * range_val  # 38.2% level
    fib_618 = prev_high - 0.618 * range_val  # 61.8% level
    
    # Align Fibonacci levels to 6t timeframe
    fib_382_aligned = align_htf_to_ltf(prices, df_1d, fib_382)
    fib_618_aligned = align_htf_to_ltf(prices, df_1d, fib_618)
    
    # Trend direction from 1d EMA50
    trend_up = close > ema_50_1d_aligned
    trend_down = close < ema_50_1d_aligned
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    # Mean reentry conditions:
    # Long: price touches or goes below 61.8% level and closes back above it (bullish rejection) in uptrend
    # Short: price touches or goes above 38.2% level and closes back below it (bearish rejection) in downtrend
    long_entry = (low <= fib_618_aligned) & (close > fib_618_aligned) & trend_up & volume_filter
    short_entry = (high >= fib_382_aligned) & (close < fib_382_aligned) & trend_down & volume_filter
    
    # Exit conditions: price reaches the opposite Fibonacci level or trend reverses
    long_exit = (high >= fib_382_aligned) | (~trend_up)
    short_exit = (low <= fib_618_aligned) | (~trend_down)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(long_entry[i]) or np.isnan(short_entry[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(volume_filter[i]) or np.isnan(fib_382_aligned[i]) or
            np.isnan(fib_618_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for mean reversion entries
            if long_entry[i]:
                signals[i] = 0.25
                position = 1
            elif short_entry[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position: exit at 38.2% level or trend reversal
            if long_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short position: exit at 61.8% level or trend reversal
            if short_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals