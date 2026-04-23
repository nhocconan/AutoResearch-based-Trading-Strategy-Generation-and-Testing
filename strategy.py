#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla Pivot Breakout with 1d EMA34 Trend Filter and Volume Confirmation
- Uses 1d Camarilla pivot levels (R3/S3 for reversal, R4/S4 for breakout)
- 1d EMA34 defines higher timeframe trend: trade R3/S3 reversals in trend direction, R4/S4 breakouts
- Volume confirmation (> 2.0x 20-period average) ensures institutional participation
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
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
    
    # Calculate 1d Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    R3 = pivot + range_1d * 1.1 / 2
    S3 = pivot - range_1d * 1.1 / 2
    R4 = pivot + range_1d * 1.1
    S4 = pivot - range_1d * 1.1
    
    # Align HTF levels to LTF
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions:
            # 1. R3 reversal: price crosses above S3 with bullish trend (close > EMA34)
            # 2. R4 breakout: price breaks above R4 with volume confirmation
            long_reversal = (close[i] > S3_aligned[i] and 
                           close[i-1] <= S3_aligned[i-1] and
                           close[i] > ema_34_1d_aligned[i])
            long_breakout = (close[i] > R4_aligned[i] and 
                           volume[i] > 2.0 * vol_ma[i])
            
            # Short conditions:
            # 1. S3 reversal: price crosses below R3 with bearish trend (close < EMA34)
            # 2. S4 breakdown: price breaks below S4 with volume confirmation
            short_reversal = (close[i] < R3_aligned[i] and 
                            close[i-1] >= R3_aligned[i-1] and
                            close[i] < ema_34_1d_aligned[i])
            short_breakout = (close[i] < S4_aligned[i] and 
                            volume[i] > 2.0 * vol_ma[i])
            
            if long_reversal or long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_reversal or short_breakout:
                signals[i] = -0.25
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
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Camarilla_R3S4_Breakout_1dEMA34_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0