#!/usr/bin/env python3
# 4H_CAMARILLA_R3_S3_BREAKOUT_1DTREND_VOLUMESPIKE
# Hypothesis: Camarilla R3/S3 breakout with 1d trend filter and volume spike filter.
# This targets 25-40 trades/year to minimize fee drag while capturing strong intraday moves
# in both bull and bear markets. The 1d trend filter ensures we trade with the higher timeframe
# direction, reducing false breakouts. Volume spike confirms institutional interest.
# Works in bull markets by capturing breakouts and in bear markets by fading false breakouts
# when price rejects at R3/S3 with volume confirmation.

name = "4H_CAMARILLA_R3_S3_BREAKOUT_1DTREND_VOLUMESPIKE"
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
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day's range
    # Using 1d data to get proper daily high/low/close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align to 4h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Calculate Camarilla levels
    range_val = prev_high_aligned - prev_low_aligned
    # R3 and S3 levels
    r3 = prev_close_aligned + range_val * 1.1 / 2
    s3 = prev_close_aligned - range_val * 1.1 / 2
    
    # 1d EMA for trend filter (34-period)
    ema1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema1d_aligned = align_htf_to_ltf(prices, df_1d, ema1d)
    
    # Volume spike filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(ema1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Close crosses above R3 with volume spike and uptrend
            if close[i] > r3[i] and close[i-1] <= r3[i-1] and vol_spike[i] and close[i] > ema1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close crosses below S3 with volume spike and downtrend
            elif close[i] < s3[i] and close[i-1] >= s3[i-1] and vol_spike[i] and close[i] < ema1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close returns below R3 or trend breaks
            if close[i] < r3[i] or close[i] < ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close returns above S3 or trend breaks
            if close[i] > s3[i] or close[i] > ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals