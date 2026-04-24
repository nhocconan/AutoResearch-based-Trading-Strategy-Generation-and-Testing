#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA20 trend filter and volume spike confirmation.
- Primary timeframe: 1h for execution, HTF: 4h for EMA trend and prior 1d OHLC for Camarilla levels.
- Camarilla pivot levels: H3 (resistance 3) and L3 (support 3) from prior 1d OHLC.
- Breakout: Close > H3 (long) or Close < L3 (short) with volume > 2.0x 20-period volume MA.
- Trend filter: Only trade breakouts in direction of 4h EMA20 (long if close > EMA20, short if close < EMA20).
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-liquidity hours.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
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
    
    # Pre-compute session hours (08:00-20:00 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA trend and 1d data for Camarilla pivots (from prior 1d OHLC)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate prior 1d OHLC for Camarilla levels (H3, L3)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    h3_1d = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low'])
    l3_1d = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low'])
    
    # Align 1d indicators to 1h (Camarilla levels based on prior 1d close)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_1d.values)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_1d.values)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 2)  # EMA20 + need at least 2 days for prior 1d OHLC
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Camarilla breakout with volume spike and trend filter
            if volume_spike[i]:
                # Long breakout: close > H3 and close > 4h EMA20 (uptrend)
                if close[i] > h3_aligned[i] and close[i] > ema_20_4h_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Short breakdown: close < L3 and close < 4h EMA20 (downtrend)
                elif close[i] < l3_aligned[i] and close[i] < ema_20_4h_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price re-enters Camarilla levels (below L3) or opposite signal
            if close[i] < l3_aligned[i]:  # Exit when price falls below L3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price re-enters Camarilla levels (above H3) or opposite signal
            if close[i] > h3_aligned[i]:  # Exit when price rises above H3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA20_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0