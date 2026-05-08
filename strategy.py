#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Camarilla_R3S3_Breakout_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1w = (close_1w > ema20_1w).astype(float)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Get daily data once for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation (R3/S3 levels)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    # Camarilla pivot levels calculation (R3 and S3)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r3 = pivot + (range_val * 1.1 / 2)  # R3 level
    s3 = pivot - (range_val * 1.1 / 2)  # S3 level
    
    # Align Camarilla levels to daily timeframe (same timeframe, no alignment needed)
    r3_1d = r3
    s3_1d = s3
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    # Price distance filter: require breakout to be at least 0.3% above/below level
    price_above_r3 = close > r3_1d * 1.003
    price_below_s3 = close < s3_1d * 0.997
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 with volume spike and weekly uptrend
            long_cond = (price_above_r3[i] and vol_spike[i] and trend_1w_aligned[i] > 0.5)
            
            # Short entry: price breaks below S3 with volume spike and weekly downtrend
            short_cond = (price_below_s3[i] and vol_spike[i] and trend_1w_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverses back below R3 (mean reversion)
            if close[i] < r3_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverses back above S3 (mean reversion)
            if close[i] > s3_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily Camarilla R3/S3 breakout with weekly EMA20 trend filter and volume confirmation.
# Targets 15-25 trades/year to avoid fee drag. Works in both bull and bear markets by
# using weekly trend filter to avoid counter-trend trades. Volume spike (2x 20-day average)
# ensures institutional participation. Price distance filter (0.3%) avoids false breakouts.
# Position size 0.25 manages drawdown. Exit at pivot level for mean reversion edge.