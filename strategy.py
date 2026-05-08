#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
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
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data once for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly EMA100 trend filter
    close_1w = df_1w['close'].values
    ema100_1w = pd.Series(close_1w).ewm(span=100, adjust=False, min_periods=100).mean().values
    trend_1w = (close_1w > ema100_1w).astype(float)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    # Camarilla pivot levels calculation (R3/S3 levels)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r3 = pivot + (range_val * 1.1 * 1.5)  # R3 = pivot + (range * 1.1 * 1.5)
    s3 = pivot - (range_val * 1.1 * 1.5)  # S3 = pivot - (range * 1.1 * 1.5)
    
    # Align Camarilla levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume spike detection: current volume > 2.0 * 50-period average
    vol_ma50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_spike = volume > (vol_ma50 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for weekly EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ma50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 with volume spike and weekly uptrend
            long_cond = (close[i] > r3_6h[i] and vol_spike[i] and trend_1w_aligned[i] > 0.5)
            
            # Short entry: price breaks below S3 with volume spike and weekly downtrend
            short_cond = (close[i] < s3_6h[i] and vol_spike[i] and trend_1w_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 (reversal signal)
            if close[i] < s3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverses back above R3 (reversal signal)
            if close[i] > r3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout strategy with volume spike confirmation and weekly EMA100 trend filter on 6h timeframe.
# Enters long when price breaks above R3 with volume spike and weekly uptrend (close > EMA100).
# Enters short when price breaks below S3 with volume spike and weekly downtrend (close < EMA100).
# Exits when price reverses back through S3/R3 respectively.
# Uses discrete sizing (0.25) to minimize churn. Targets 15-25 trades/year on 6h timeframe.
# Weekly trend filter ensures we only trade with the higher timeframe trend, reducing whipsaw in sideways markets.
# Works in bull markets (trend-following breakouts) and bear markets (reversal breakouts from overextended levels).