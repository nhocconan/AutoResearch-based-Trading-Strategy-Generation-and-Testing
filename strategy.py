#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H4/L4 breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1w for EMA trend and Camarilla levels.
- Camarilla pivot levels: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2.
- Entry: Long when price breaks above H4 with volume spike and price > 1w EMA50 (uptrend).
         Short when price breaks below L4 with volume spike and price < 1w EMA50 (downtrend).
- Exit: When price reverts to 1w EMA50 or opposite signal.
- Uses weekly timeframe to reduce noise and capture major trend shifts.
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
    
    # Get 1w data for Camarilla levels and EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels on 1w
    # H4 = close + 1.1*(high-low)/2
    # L4 = close - 1.1*(high-low)/2
    camarilla_h4 = df_1w['close'].values + (1.1 * (df_1w['high'].values - df_1w['low'].values) / 2)
    camarilla_l4 = df_1w['close'].values - (1.1 * (df_1w['high'].values - df_1w['low'].values) / 2)
    
    # Align 1w indicators to 6h
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough 1w bars for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Camarilla breakout signals with volume spike and trend filter
            if volume_spike[i]:
                # Long: price breaks above H4 in uptrend
                if close[i] > camarilla_h4_aligned[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below L4 in downtrend
                elif close[i] < camarilla_l4_aligned[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price reverts to EMA50 or short signal
            if close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to EMA50 or long signal
            if close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H4L4_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0