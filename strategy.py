#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R3S3_Breakout_1dTrend_4hVolume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 4h data for volume filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d = (close_1d > ema34_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
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
    
    # Align Camarilla levels to 1h timeframe
    r3_1h = align_htf_to_ltf(prices, df_1d, r3)
    s3_1h = align_htf_to_ltf(prices, df_1d, s3)
    
    # 4h volume filter: current 4h volume > 2.0 * 20-period average
    vol_4h = df_4h['volume'].values
    vol_ma20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike_4h = vol_4h > (vol_ma20 * 2.0)
    vol_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_spike_4h)
    
    # Session filter: 08-20 UTC (already datetime64)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_1h[i]) or np.isnan(s3_1h[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_spike_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 with 4h volume spike and 1d uptrend, in session
            long_cond = (close[i] > r3_1h[i] and vol_spike_4h_aligned[i] and 
                        trend_1d_aligned[i] > 0.5 and session_filter[i])
            
            # Short entry: price breaks below S3 with 4h volume spike and 1d downtrend, in session
            short_cond = (close[i] < s3_1h[i] and vol_spike_4h_aligned[i] and 
                         trend_1d_aligned[i] < 0.5 and session_filter[i])
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 (reversal signal)
            if close[i] < s3_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price reverses back above R3 (reversal signal)
            if close[i] > r3_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Camarilla R3/S3 breakout strategy with 1d EMA34 trend filter and 4h volume spike confirmation.
# Uses 1h for precise entry timing while relying on 1d trend and 4h volume for signal quality.
# Session filter (08-20 UTC) reduces noise during low-liquidity hours.
# Targets 15-35 trades/year by requiring multiple confluence factors.
# Works in bull markets (trend-following breakouts) and bear markets (reversal from extreme levels).
# Discrete sizing (0.20) minimizes turnover and fee drag.