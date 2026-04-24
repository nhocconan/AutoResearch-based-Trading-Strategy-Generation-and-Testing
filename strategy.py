#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 12h EMA trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 12h for EMA trend direction.
- EMA50 > EMA200 on 12h indicates bullish trend (favor longs), EMA50 < EMA200 indicates bearish trend (favor shorts).
- Entry: Long when price breaks above Camarilla R3 AND EMA50 > EMA200 (bullish breakout in bull trend).
         Short when price breaks below Camarilla S3 AND EMA50 < EMA200 (bearish breakout in bear trend).
- Exit: Opposite Camarilla breakout (R4/S4) or EMA trend flip.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-150 total trades over 4 years (19-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMAs (50 and 200) on 12h
    close_12h = pd.Series(df_12h['close'].values)
    ema50 = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200 = close_12h.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h EMAs to 6h
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50)
    ema200_aligned = align_htf_to_ltf(prices, df_12h, ema200)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # Typical price = (high + low + close) / 3
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3.0
    typical_price_values = typical_price.values
    
    # Camarilla levels based on previous bar's range
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_prev = df_12h['close'].values
    
    # Pivot point (standard)
    pivot = (high_12h + low_12h + close_12h_prev) / 3.0
    # Range
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    r3 = pivot + (range_12h * 1.1 / 4.0)  # R3
    s3 = pivot - (range_12h * 1.1 / 4.0)  # S3
    r4 = pivot + (range_12h * 1.1 / 2.0)  # R4 (exit for longs)
    s4 = pivot - (range_12h * 1.1 / 2.0)  # S4 (exit for shorts)
    
    # Align Camarilla levels to 6h
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 200, 20)  # Need enough 12h bars for EMA200 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(ema200_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        ema50_val = ema50_aligned[i]
        ema200_val = ema200_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Bullish breakout: price breaks above R3 AND EMA50 > EMA200 (bull trend)
                if curr_high > r3_aligned[i] and ema50_val > ema200_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below S3 AND EMA50 < EMA200 (bear trend)
                elif curr_low < s3_aligned[i] and ema50_val < ema200_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks above R4 (stop and reverse) OR EMA trend flips to bearish
            if curr_high > r4_aligned[i] or ema50_val < ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks below S4 (stop and reverse) OR EMA trend flips to bullish
            if curr_low < s4_aligned[i] or ema50_val > ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_12hEMATrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0