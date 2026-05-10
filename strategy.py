#!/usr/bin/env python3
# 4h_Camarilla_R3S3_Breakout_1dTrend_Only
# Hypothesis: Uses Camarilla pivot levels (R3/S3) from 1d timeframe for breakout signals.
# Goes long when price breaks above R3 with 1d uptrend (price > 1d EMA34).
# Goes short when price breaks below S3 with 1d downtrend (price < 1d EMA34).
# Removes volume confirmation to reduce trade frequency and avoid overtrading.
# Focuses on clean breakouts with higher timeframe trend alignment.
# Targets 15-30 trades per year on 4h timeframe with position size 0.25.
# Uses 1d EMA(34) for trend filter to avoid counter-trend trades.

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Only"
timeframe = "4h"
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
    
    # Get 1d data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R3 = Close + (High - Low) * 1.1/2
    # S3 = Close - (High - Low) * 1.1/2
    # Using previous day's data to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate R3 and S3
    r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Warmup for 1d EMA
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema_34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 1d
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R3 with 1d uptrend
            if close[i] > r3_aligned[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 with 1d downtrend
            elif close[i] < s3_aligned[i] and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below R3 or trend turns down
            if close[i] < r3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above S3 or trend turns up
            if close[i] > s3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals