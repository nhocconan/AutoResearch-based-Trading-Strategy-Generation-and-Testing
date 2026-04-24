#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for Camarilla pivots, EMA trend, and volume context.
- Camarilla pivots: Calculate R3, R4, S3, S4 from prior 1d OHLC.
- Entry: Long when price breaks above R4 with volume spike and close > 1d EMA34 (strong uptrend continuation).
         Short when price breaks below S3 with volume spike and close < 1d EMA34 (strong downtrend continuation).
- Exit: When price reverts to the 1d VWAP (mean reversion to fair value) or opposite signal.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots, EMA trend, and VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d VWAP for exit (typical price * volume cumsum)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap = vwap.values
    
    # Calculate Camarilla pivot levels from prior 1d OHLC
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.1 * (High - Low)
    # S3 = Close - 1.1 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    camarilla_r4 = df_1d['close'] + 1.5 * (df_1d['high'] - df_1d['low'])
    camarilla_r3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low'])
    camarilla_s3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low'])
    camarilla_s4 = df_1d['close'] - 1.5 * (df_1d['high'] - df_1d['low'])
    
    # Align 1d indicators to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4.values)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4.values)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(vwap_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for breakout signals with volume spike and trend filter
            if volume_spike[i]:
                # Long: price breaks above R4 in uptrend
                if close[i] > camarilla_r4_aligned[i] and close[i] > ema_34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below S3 in downtrend
                elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_34_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to VWAP or opposite signal
            if close[i] <= vwap_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to VWAP or opposite signal
            if close[i] >= vwap_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S4_Breakout_1dEMA34_VWAPExit_v1"
timeframe = "6h"
leverage = 1.0