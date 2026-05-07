# 6h Camarilla Pivot R3/S3 Breakout with 1d Trend and Volume Spike
# Hypothesis: Camarilla pivot levels act as strong support/resistance. Breakouts above R3 or below S3
# with volume confirmation and aligned with the 1d trend capture momentum moves. This strategy works
# in both bull and bear markets by following the daily trend direction while using pivot levels
# for precise entry/exit points. Target: 20-40 trades/year (80-160 total over 4 years).
# Uses discrete position sizing (0.25) to minimize fee churn.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # Camarilla pivot levels from previous day (using typical price)
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # For intraday, we use previous day's typical price to calculate pivots
    # We'll calculate pivots using a rolling window of previous day's data
    # Since we're on 6h timeframe, 4 bars = 1 day
    tp_series = pd.Series(typical_price)
    # Previous day's typical price (4 bars back)
    prev_day_tp = tp_series.shift(4)
    # For pivot calculation, we need the previous day's high, low, close
    prev_day_high = pd.Series(high).shift(4)
    prev_day_low = pd.Series(low).shift(4)
    prev_day_close = pd.Series(close).shift(4)
    
    # Pivot point = (prev_high + prev_low + prev_close) / 3
    pivot = (prev_day_high + prev_day_low + prev_day_close) / 3.0
    # Camarilla levels
    range_val = prev_day_high - prev_day_low
    r3 = pivot + (range_val * 1.1 / 2)
    s3 = pivot - (range_val * 1.1 / 2)
    r4 = pivot + (range_val * 1.1)
    s4 = pivot - (range_val * 1.1)
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 2.0x 24-period average (4 days on 6h)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 34)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(pivot[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(r4[i]) or 
            np.isnan(s4[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(ema34_rising[i]) or np.isnan(ema34_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3, 1d EMA34 rising, volume filter
            long_cond = (close[i] > r3[i]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: price breaks below S3, 1d EMA34 falling, volume filter
            short_cond = (close[i] < s3[i]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below R3 (or optionally at R4 for profit taking)
            if close[i] < r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above S3 (or optionally at S4 for profit taking)
            if close[i] > s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals