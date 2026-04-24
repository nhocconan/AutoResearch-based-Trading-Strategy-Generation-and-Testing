#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for EMA trend and Camarilla levels.
- Camarilla pivot levels: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2 (based on prior 1d candle).
- Entry: Long when price breaks above H3 with volume spike and price > 1d EMA34 (uptrend).
         Short when price breaks below L3 with volume spike and price < 1d EMA34 (downtrend).
- Exit: When price reverts to 1d close (pivot point) or opposite signal.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
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
    
    # Get 1d data for Camarilla and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels H3 and L3 from prior 1d candle
    # H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    camarilla_h3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 2
    camarilla_l3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 2
    camarilla_pivot = df_1d['close']  # pivot point for exit
    
    # Align 1d indicators to 12h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3.values)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3.values)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot.values)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Camarilla breakout signals with volume spike and trend filter
            if volume_spike[i]:
                # Long: price breaks above H3 in uptrend
                if close[i] > camarilla_h3_aligned[i] and close[i] > ema_34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below L3 in downtrend
                elif close[i] < camarilla_l3_aligned[i] and close[i] < ema_34_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to pivot point or opposite signal
            if close[i] <= camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to pivot point or opposite signal
            if close[i] >= camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0