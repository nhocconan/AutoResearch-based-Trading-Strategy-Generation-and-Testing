#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Fibonacci pivot levels from previous weekly bar
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_w = prev_high - prev_low
    
    # Weekly Fibonacci levels (0.382, 0.618)
    fib_r2 = pivot + 0.618 * range_w
    fib_s2 = pivot - 0.618 * range_w
    fib_r1 = pivot + 0.382 * range_w
    fib_s1 = pivot - 0.382 * range_w
    
    # Align Fibonacci levels to 6h timeframe
    fib_r2_aligned = align_htf_to_ltf(prices, df_1w, fib_r2)
    fib_s2_aligned = align_htf_to_ltf(prices, df_1w, fib_s2)
    fib_r1_aligned = align_htf_to_ltf(prices, df_1w, fib_r1)
    fib_s1_aligned = align_htf_to_ltf(prices, df_1w, fib_s1)
    
    # Weekly EMA trend filter (21-period)
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume filter: volume > 2 x 28-period average (6h periods = 7 days)
    vol_ma_28 = np.full(n, np.nan)
    for i in range(27, n):
        vol_ma_28[i] = np.mean(volume[i-27:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly pivot (1 week), EMA (21), volume MA (28)
    start_idx = max(1, 21, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(fib_r2_aligned[i]) or np.isnan(fib_s2_aligned[i]) or
            np.isnan(fib_r1_aligned[i]) or np.isnan(fib_s1_aligned[i]) or
            np.isnan(ema_21_aligned[i]) or np.isnan(vol_ma_28[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_28[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Trend filter from weekly EMA
        bullish_trend = price > ema_21_aligned[i]
        bearish_trend = price < ema_21_aligned[i]
        
        fib_r2 = fib_r2_aligned[i]
        fib_s2 = fib_s2_aligned[i]
        fib_r1 = fib_r1_aligned[i]
        fib_s1 = fib_s1_aligned[i]
        
        if position == 0:
            # Long: price breaks above Fib R2 + volume + bullish weekly trend
            if price > fib_r2 and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below Fib S2 + volume + bearish weekly trend
            elif price < fib_s2 and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Fib S1 or trend turns bearish
            if price < fib_s1 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Fib R1 or trend turns bullish
            if price > fib_r1 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Fibonacci_R2S2_Breakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0