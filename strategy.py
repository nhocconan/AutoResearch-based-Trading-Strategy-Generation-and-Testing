#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla Pivot Breakout with 4h EMA50 Trend Filter and Volume Confirmation
- Uses 4h Camarilla pivot levels (R3/S3 for reversal, R4/S4 for breakout)
- 4h EMA50 defines higher timeframe trend: trade R3/S3 reversals in trend direction, R4/S4 breakouts
- Volume confirmation (> 1.8x 20-period average) ensures institutional participation
- Session filter (08-20 UTC) reduces noise trades outside active market hours
- Designed for 1h timeframe targeting 15-37 trades/year (60-150 over 4 years)
- Works in both bull and bear markets by combining mean reversion (R3/S3) and breakout (R4/S4) logic
"""

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
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h Camarilla pivot levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla calculations
    pivot = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    
    R3 = pivot + range_4h * 1.1 / 2
    S3 = pivot - range_4h * 1.1 / 2
    R4 = pivot + range_4h * 1.1
    S4 = pivot - range_4h * 1.1
    
    # Align HTF levels to LTF
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3)
    R4_aligned = align_htf_to_ltf(prices, df_4h, R4)
    S4_aligned = align_htf_to_ltf(prices, df_4h, S4)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions:
            # 1. R3 reversal: price crosses above S3 with bullish trend (close > EMA50)
            # 2. R4 breakout: price breaks above R4 with volume confirmation
            long_reversal = (close[i] > S3_aligned[i] and 
                           close[i-1] <= S3_aligned[i-1] and
                           close[i] > ema_50_4h_aligned[i])
            long_breakout = (close[i] > R4_aligned[i] and 
                           volume[i] > 1.8 * vol_ma[i])
            
            # Short conditions:
            # 1. S3 reversal: price crosses below R3 with bearish trend (close < EMA50)
            # 2. S4 breakdown: price breaks below S4 with volume confirmation
            short_reversal = (close[i] < R3_aligned[i] and 
                            close[i-1] >= R3_aligned[i-1] and
                            close[i] < ema_50_4h_aligned[i])
            short_breakout = (close[i] < S4_aligned[i] and 
                            volume[i] > 1.8 * vol_ma[i])
            
            if long_reversal or long_breakout:
                signals[i] = 0.20
                position = 1
            elif short_reversal or short_breakout:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below S3 (reversal fail) or crosses above R4 (take profit)
                if (close[i] < S3_aligned[i] and close[i-1] >= S3_aligned[i-1]) or \
                   (close[i] > R4_aligned[i] and close[i-1] <= R4_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above R3 (reversal fail) or crosses below S4 (take profit)
                if (close[i] > R3_aligned[i] and close[i-1] <= R3_aligned[i-1]) or \
                   (close[i] < S4_aligned[i] and close[i-1] >= S4_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Camarilla_R3S4_Breakout_4hEMA50_Trend_VolumeConfirm_Session"
timeframe = "1h"
leverage = 1.0