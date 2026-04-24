#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for EMA trend and Camarilla levels.
- Camarilla pivot levels: H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6.
- Entry: Long when price breaks above H3 with volume spike and price > 1d EMA34 (uptrend).
         Short when price breaks below L3 with volume spike and price < 1d EMA34 (downtrend).
- Exit: When price reverts to 1d EMA34 or opposite signal.
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
    
    # Get 1d data for Camarilla levels and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels on 1d
    # H3 = close + 1.1*(high-low)/6
    # L3 = close - 1.1*(high-low)/6
    camarilla_h3 = df_1d['close'].values + (1.1 * (df_1d['high'].values - df_1d['low'].values) / 6)
    camarilla_l3 = df_1d['close'].values - (1.1 * (df_1d['high'].values - df_1d['low'].values) / 6)
    
    # Align 1d indicators to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough 1d bars for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_spike[i])):
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
            # Long exit: price reverts to EMA34 or short signal
            if close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to EMA34 or long signal
            if close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0