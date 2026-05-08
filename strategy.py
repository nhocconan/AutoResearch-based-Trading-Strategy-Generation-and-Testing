#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume_Spike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_open = np.roll(df_1d['open'].values, 1)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    prev_open[0] = df_1d['open'].values[0]
    
    # Camarilla pivot levels calculation
    # Pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # Range
    range_val = prev_high - prev_low
    # Resistance and Support levels
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    r2 = pivot + (range_val * 1.1 / 6)
    s2 = pivot - (range_val * 1.1 / 6)
    r3 = pivot + (range_val * 1.1 / 4)
    s3 = pivot - (range_val * 1.1 / 4)
    r4 = pivot + (range_val * 1.1 / 2)
    s4 = pivot - (range_val * 1.1 / 2)
    
    # Align all levels to 4h timeframe
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2)
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    r4_4h = align_htf_to_ltf(prices, df_1d, r4)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(r2_4h[i]) or np.isnan(s2_4h[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 with volume spike and above 12h EMA50 (uptrend)
            long_cond = (close[i] > r1_4h[i] and 
                        vol_spike[i] and 
                        close[i] > ema50_12h_aligned[i])
            
            # Short entry: price breaks below S1 with volume spike and below 12h EMA50 (downtrend)
            short_cond = (close[i] < s1_4h[i] and 
                         vol_spike[i] and 
                         close[i] < ema50_12h_aligned[i])
            
            if long_cond:
                signals[i] = 0.30
                position = 1
            elif short_cond:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 (reversal signal)
            if close[i] < s1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price breaks above R1 (reversal signal)
            if close[i] > r1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: Camarilla R1/S1 breakout strategy with 12h EMA50 trend filter and volume spike confirmation.
# Enters long when price breaks above R1 with volume spike in uptrend (price > 12h EMA50).
# Enters short when price breaks below S1 with volume spike in downtrend (price < 12h EMA50).
# Exits when price reverses back through S1/R1 respectively.
# Uses discrete sizing (0.30) to minimize churn. Designed for 4h timeframe to target 20-50 trades/year.
# Works in both bull markets (trend following breaks) and bear markets (mean reversion fails, trend filters prevent false signals).