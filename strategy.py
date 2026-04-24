#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1h for execution, HTF: 4h for EMA trend and prior 1d OHLC for Camarilla levels.
- Camarilla pivot levels: H3 (resistance 3) and L3 (support 3) from prior 4h OHLC (proxy for prior 1d structure).
- Breakout: Close > H3 (long) or Close < L3 (short) with volume > 2.0x 20-period volume MA.
- Trend filter: Only trade breakouts in direction of 4h EMA50 (long if close > EMA50, short if close < EMA50).
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-volume Asian session noise.
- Discrete signal size: 0.20 to limit drawdown and reduce fee churn.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
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
    
    # Get 4h data for Camarilla pivots and EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate prior 4h OHLC for Camarilla levels (H3, L3)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    h3 = df_4h['close'] + 1.1 * (df_4h['high'] - df_4h['low'])
    l3 = df_4h['close'] - 1.1 * (df_4h['high'] - df_4h['low'])
    
    # Calculate 4h EMA50 for trend filter
    ema_50 = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h indicators to 1h
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3.values)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Camarilla breakout with volume spike and trend filter
            if volume_spike[i]:
                # Long breakout: close > H3 and close > 4h EMA50 (uptrend)
                if close[i] > h3_aligned[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Short breakdown: close < L3 and close < 4h EMA50 (downtrend)
                elif close[i] < l3_aligned[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price re-enters Camarilla levels or opposite signal
            if close[i] < l3_aligned[i]:  # Exit when price falls below L3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price re-enters Camarilla levels or opposite signal
            if close[i] > h3_aligned[i]:  # Exit when price rises above H3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0