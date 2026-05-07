#!/usr/bin/env python3
# 12h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot breakouts on 12h with daily trend filter and volume confirmation capture institutional moves in both bull and bear markets.
# Timeframe: 12h, uses 1d trend filter for multi-timeframe alignment.
# Low trade frequency (~15-25/year) via strict R3/S3 breakout + volume + trend confluence.
# Long: Breakout above R3 with volume > 1.5x average and daily uptrend.
# Short: Breakdown below S3 with volume > 1.5x average and daily downtrend.
# Exit: Opposite Camarilla level (S3 for long, R3 for short) or trend failure.

timeframe = "12h"
name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # Calculate Camarilla levels from previous day
    # Using previous day's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    R3 = pivot + (range_hl * 1.1 / 2)
    S3 = pivot - (range_hl * 1.1 / 2)
    
    # Average volume for spike detection (2-period = 1 day on 12h chart)
    vol_ma = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    # Get 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(2, 34)  # Ensure we have Camarilla, volume MA and EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above R3 with volume spike and daily uptrend
            if close[i] > R3[i] and volume[i] > 1.5 * vol_ma[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S3 with volume spike and daily downtrend
            elif close[i] < S3[i] and volume[i] > 1.5 * vol_ma[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: breakdown below S3 or trend failure
            if close[i] < S3[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: breakout above R3 or trend failure
            if close[i] > R3[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals