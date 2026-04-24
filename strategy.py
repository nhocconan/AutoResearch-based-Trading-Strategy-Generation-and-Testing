#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume spike confirmation.
- Uses 1h timeframe (primary) and 4h HTF for EMA50 trend alignment (proven BTC/ETH edge from DB).
- Camarilla pivot levels calculated from prior 1d high/low/close.
- Breakout logic: long when price closes above H3 with volume spike and uptrend,
                  short when price closes below L3 with volume spike and downtrend.
- Trend filter: only long when 1h close > 4h EMA50, only short when 1h close < 4h EMA50.
- Volume confirmation: current 1h volume > 2.0 * 20-period 1h volume MA (strict to reduce trades).
- Discrete signal size: 0.20 to balance reward and risk, minimizing fee churn.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
- Session filter: only trade between 08:00-20:00 UTC to avoid low-liquidity hours.
- Works in both bull/bear: trend filter avoids counter-trend trades, Camarilla breakouts capture momentum in all regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from prior 1d OHLC (using mtf_data for 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Prior 1d OHLC (shifted by 1 to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Align 1d data to 1h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Camarilla calculations (based on prior 1d range)
    camarilla_range = high_1d_aligned - low_1d_aligned
    camarilla_H3 = close_1d_aligned + camarilla_range * 1.1 / 4
    camarilla_L3 = close_1d_aligned - camarilla_range * 1.1 / 4
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (strict)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Trend filter: 1h close vs 4h EMA50
    uptrend = close > ema_50_4h_aligned
    downtrend = close < ema_50_4h_aligned
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Need 4h EMA50 and sufficient volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_H3[i]) or 
            np.isnan(camarilla_L3[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if outside session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above H3 AND uptrend AND volume spike
            if close[i] > camarilla_H3[i] and uptrend[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: price closes below L3 AND downtrend AND volume spike
            elif close[i] < camarilla_L3[i] and downtrend[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price reverts to midpoint of Camarilla levels or reverse signal
            camarilla_mid = (camarilla_H3[i] + camarilla_L3[i]) / 2
            if close[i] <= camarilla_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price reverts to midpoint of Camarilla levels or reverse signal
            camarilla_mid = (camarilla_H3[i] + camarilla_L3[i]) / 2
            if close[i] >= camarilla_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3_L3_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0