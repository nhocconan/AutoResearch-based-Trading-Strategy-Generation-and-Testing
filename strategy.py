#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Camarilla pivot levels (R1/S1) from 1-day timeframe provide precise entry/exit points.
# Weekly trend filter (1w EMA34) ensures we trade with the higher timeframe trend.
# Volume confirmation filters out low-liquidity breakouts.
# Designed for low trade frequency (12-37/year) to minimize fee drag on 12h timeframe.
# Works in bull markets via buying R1 breakouts in uptrend and selling S1 breakdowns in downtrend.

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC data"""
    # Typical price
    typical = (high + low + close) / 3
    # Range
    range_val = high - low
    # Camarilla levels
    R4 = close + range_val * 1.1 / 2
    R3 = close + range_val * 1.1 / 4
    R2 = close + range_val * 1.1 / 6
    R1 = close + range_val * 1.1 / 12
    S1 = close - range_val * 1.1 / 12
    S2 = close - range_val * 1.1 / 6
    S3 = close - range_val * 1.1 / 4
    S4 = close - range_val * 1.1 / 2
    return R1, S1, R2, S2, R3, S3, R4, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's data
    # Shift by 1 to use only completed daily data (no look-ahead)
    high_shift = df_1d['high'].shift(1).values
    low_shift = df_1d['low'].shift(1).values
    close_shift = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    R1, S1, R2, S2, R3, S3, R4, S4 = calculate_camarilla(high_shift, low_shift, close_shift)
    
    # Align Camarilla levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Get 12h data for entry signals
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA (34) + volume EMA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from weekly EMA34
        # Uptrend: price above EMA34, Downtrend: price below EMA34
        trend_up = close[i] > ema_34_1w_aligned[i]
        trend_down = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R1 AND weekly uptrend AND volume confirmation
            if close[i] > R1_aligned[i] and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 AND weekly downtrend AND volume confirmation
            elif close[i] < S1_aligned[i] and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price falls below S1 (reversal signal) OR weekly trend turns down
            if close[i] < S1_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price rises above R1 (reversal signal) OR weekly trend turns up
            if close[i] > R1_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals