#!/usr/bin/env python3
"""
Hypothesis: 6h Weekly Camarilla Pivot Breakout with 1d Volume Confirmation and ATR Filter
- Uses weekly Camarilla pivot levels (R3, S3, R4, S4) from 1w timeframe for structure
- Breakout above R3 or below S3 with confirmation (close outside level + volume spike)
- Continuation breakout at R4/S4 for stronger moves
- 1d ATR filter: only trade when ATR(14) > 0.5 * ATR(50) to avoid low-volatility chop
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Weekly pivots provide significant support/resistance that works in both bull and bear markets
- Volume confirmation filters false breakouts; ATR filter avoids ranging markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # Need at least 5 periods for reliable pivots
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivots for each weekly bar
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    r3 = pivot + (range_1w * 1.1 / 4)
    s3 = pivot - (range_1w * 1.1 / 4)
    r4 = pivot + (range_1w * 1.1 / 2)
    s4 = pivot - (range_1w * 1.1 / 2)
    
    # Align weekly Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Calculate 1d ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need 50 periods for ATR(50)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original index
    
    # ATR using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def atr_wilder(high, low, close, period):
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        atr = np.full_like(tr, np.nan)
        if len(tr) < period:
            return atr
        # First value is simple average
        atr[period-1] = np.nanmean(tr[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_14 = atr_wilder(high_1d, low_1d, close_1d, 14)
    atr_50 = atr_wilder(high_1d, low_1d, close_1d, 50)
    
    # Align ATR ratios to 6h timeframe
    atr_ratio = atr_14 / atr_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(5, 50, 20)  # Weekly pivots, ATR(50), volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR(14) > 0.5 * ATR(50)
        volatile_enough = atr_ratio_aligned[i] > 0.5
        
        if position == 0:
            # Long breakout conditions
            long_breakout_r3 = close[i] > r3_aligned[i] and volume[i] > 1.3 * vol_ma[i]
            long_breakout_r4 = close[i] > r4_aligned[i] and volume[i] > 1.3 * vol_ma[i]
            
            # Short breakout conditions
            short_breakout_s3 = close[i] < s3_aligned[i] and volume[i] > 1.3 * vol_ma[i]
            short_breakout_s4 = close[i] < s4_aligned[i] and volume[i] > 1.3 * vol_ma[i]
            
            # Enter long on R3/R4 breakout with volume and volatility
            if (long_breakout_r3 or long_breakout_r4) and volatile_enough:
                signals[i] = 0.25
                position = 1
            # Enter short on S3/S4 breakout with volume and volatility
            elif (short_breakout_s3 or short_breakout_s4) and volatile_enough:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite breakout or loss of momentum
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below S3 or weak volume
                if close[i] < s3_aligned[i] or volume[i] < 0.7 * vol_ma[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above R3 or weak volume
                if close[i] > r3_aligned[i] or volume[i] < 0.7 * vol_ma[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Weekly_Camarilla_R3S3_Breakout_1dATR_VolumeFilter"
timeframe = "6h"
leverage = 1.0