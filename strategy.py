#!/usr/bin/env python3
# 12h_Camarilla_Pullback_With_1dTrend
# Hypothesis: On 12h timeframe, price often pulls back to Camarilla pivot levels (S3/S4 for long, R3/R4 for short) during strong daily trends.
# Enter when price touches S3 (long) or R3 (short) on 12h chart while 1d trend is confirmed (close > EMA50 for long, close < EMA50 for short).
# Use volume spike (1.5x 24-period MA) to confirm momentum and avoid false breakouts.
# Exit when price reverses to opposite Camarilla level (S4 for long exit, R4 for short exit) or trend breaks.
# Works in bull markets (follows uptrend pullbacks to S3) and bear markets (follows downtrend pullbacks to R3).
# Low trade frequency due to strict confluence: trend + pivot touch + volume.

name = "12h_Camarilla_Pullback_With_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12-period EMA on 12h for trend filter (faster than 50-period)
    close_s = pd.Series(close)
    ema_12 = close_s.ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous day
    # Camarilla: based on previous day's range
    # Resistance levels: R3 = close + 1.1*(high-low)/2, R4 = close + 1.1*(high-low)
    # Support levels: S3 = close - 1.1*(high-low)/2, S4 = close - 1.1*(high-low)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    r3 = prev_close + 1.1 * prev_range / 2
    r4 = prev_close + 1.1 * prev_range
    s3 = prev_close - 1.1 * prev_range / 2
    s4 = prev_close - 1.1 * prev_range
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation (24-period MA on 12h = 12 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA12 (12), EMA50_1d (50), volume MA (24), Camarilla (need prev day)
    start_idx = max(12, 50, 24) + 1  # +1 for previous day shift
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_12[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter: use 12h EMA vs daily EMA50 for smoother filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Price touching Camarilla levels with wick close
        # For long: touch S3 from above (wick low <= S3) and close above S3
        # For short: touch R3 from below (wick high >= R3) and close below R3
        touch_s3 = low[i] <= s3_aligned[i] and close[i] > s3_aligned[i]
        touch_r3 = high[i] >= r3_aligned[i] and close[i] < r3_aligned[i]
        
        if position == 0:
            # Long entry: uptrend + touch S3 + volume
            if uptrend and touch_s3 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + touch R3 + volume
            elif downtrend and touch_r3 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or touch S4 (stronger reversal)
            if not uptrend or low[i] <= s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or touch R4 (stronger reversal)
            if not downtrend or high[i] >= r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals