#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1d = close_1d > ema34_1d
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    # Calculate Camarilla levels from previous 1d
    # H, L, C from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid division by zero and NaN
    hlc = prev_high - prev_low
    hlc_safe = np.where(hlc == 0, 1e-10, hlc)
    
    # Camarilla levels
    R3 = prev_close + (hlc_safe * 1.1 / 4)
    R2 = prev_close + (hlc_safe * 1.1 / 6)
    R1 = prev_close + (hlc_safe * 1.1 / 12)
    S1 = prev_close - (hlc_safe * 1.1 / 12)
    S2 = prev_close - (hlc_safe * 1.1 / 6)
    S3 = prev_close - (hlc_safe * 1.1 / 4)
    
    # Align levels to 4h (they are constant throughout the day)
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    R2_4h = align_htf_to_ltf(prices, df_1d, R2)
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    S2_4h = align_htf_to_ltf(prices, df_1d, S2)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(R3_4h[i]) or np.isnan(R2_4h[i]) or np.isnan(R1_4h[i]) or
            np.isnan(S1_4h[i]) or np.isnan(S2_4h[i]) or np.isnan(S3_4h[i]) or
            np.isnan(trend_up_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above R3 with daily uptrend + volume confirmation
            if close[i] > R3_4h[i] and trend_up_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with daily downtrend + volume confirmation
            elif close[i] < S3_4h[i] and not trend_up_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price closes below R1 or daily trend turns down
            if close[i] < R1_4h[i] or not trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price closes above S1 or daily trend turns up
            if close[i] > S1_4h[i] or trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals