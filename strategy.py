#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1h for execution, HTF: 4h for EMA trend and Camarilla levels.
- Camarilla pivot levels: H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6.
- Entry: Long when price breaks above H3 with volume spike and price > 4h EMA50 (uptrend).
         Short when price breaks below L3 with volume spike and price < 4h EMA50 (downtrend).
- Exit: When price reverts to 4h EMA50 or opposite signal.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-liquidity hours.
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
    
    # Get 4h data for Camarilla levels and EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50 = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels on 4h
    # H3 = close + 1.1*(high-low)/6
    # L3 = close - 1.1*(high-low)/6
    camarilla_h3 = df_4h['close'].values + (1.1 * (df_4h['high'].values - df_4h['low'].values) / 6)
    camarilla_l3 = df_4h['close'].values - (1.1 * (df_4h['high'].values - df_4h['low'].values) / 6)
    
    # Align 4h indicators to 1h
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 1h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough 4h bars for EMA50
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i] or \
           (np.isnan(ema_50_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Camarilla breakout signals with volume spike and trend filter
            if volume_spike[i]:
                # Long: price breaks above H3 in uptrend
                if close[i] > camarilla_h3_aligned[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Short: price breaks below L3 in downtrend
                elif close[i] < camarilla_l3_aligned[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price reverts to EMA50 or short signal
            if close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price reverts to EMA50 or long signal
            if close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0