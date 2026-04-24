#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1w EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h for execution, HTF: 1w for EMA trend.
- Camarilla pivot levels: H3 (resistance 3) and L3 (support 3) from prior 1w OHLC.
- Breakout: Close > H3 (long) or Close < L3 (short) with volume > 2.0x 20-period volume MA.
- Trend filter: Only trade breakouts in direction of 1w EMA50 (long if close > EMA50, short if close < EMA50).
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
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
    
    # Get 1w data for Camarilla pivots and EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate prior 1w OHLC for Camarilla levels (H3, L3)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    h3 = df_1w['close'] + 1.1 * (df_1w['high'] - df_1w['low'])
    l3 = df_1w['close'] - 1.1 * (df_1w['high'] - df_1w['low'])
    
    # Calculate 1w EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w indicators to 4h
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3.values)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Camarilla breakout with volume spike and trend filter
            if volume_spike[i]:
                # Long breakout: close > H3 and close > 1w EMA50 (uptrend)
                if close[i] > h3_aligned[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown: close < L3 and close < 1w EMA50 (downtrend)
                elif close[i] < l3_aligned[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price re-enters Camarilla levels or opposite signal
            if close[i] < l3_aligned[i]:  # Exit when price falls below L3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters Camarilla levels or opposite signal
            if close[i] > h3_aligned[i]:  # Exit when price rises above H3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0