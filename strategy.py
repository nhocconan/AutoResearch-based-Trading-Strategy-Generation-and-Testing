#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA34 trend filter and volume confirmation.
- Uses 4h timeframe (primary) and 12h HTF for EMA34 trend alignment
- Camarilla levels calculated from prior 1d OHLC: H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4
- Breakout logic: long when price closes above H3 with volume spike and uptrend,
                  short when price closes below L3 with volume spike and downtrend
- Trend filter: only long when 4h close > 12h EMA34, only short when 4h close < 12h EMA34
- Volume confirmation: current 4h volume > 1.5 * 20-period 4h volume MA
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Camarilla breakouts capture momentum in all regimes
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
    
    # Calculate 4h close for trend filter (using close vs EMA)
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate prior 1d Camarilla levels (H3, L3)
    # Need to shift 1d data by 1 to avoid look-ahead (use prior completed day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's Camarilla: H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4
    camarilla_h3_1d = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3_1d = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align to 4h timeframe (wait for 1d bar to close)
    camarilla_h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    # Trend filter: 4h close vs 12h EMA34
    uptrend = close > ema_34_12h_aligned
    downtrend = close < ema_34_12h_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 34)  # Need 12h EMA34 and sufficient volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(camarilla_h3_1d_aligned[i]) or 
            np.isnan(camarilla_l3_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above H3 AND uptrend AND volume spike
            if close[i] > camarilla_h3_1d_aligned[i] and uptrend[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below L3 AND downtrend AND volume spike
            elif close[i] < camarilla_l3_1d_aligned[i] and downtrend[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to prior 1d L4 (mean reversion) or reverse signal
            camarilla_l4_1d = close_1d - 1.1 * (high_1d - low_1d) / 2
            camarilla_l4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
            if not np.isnan(camarilla_l4_1d_aligned[i]) and close[i] <= camarilla_l4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to prior 1d H4 (mean reversion) or reverse signal
            camarilla_h4_1d = close_1d + 1.1 * (high_1d - low_1d) / 2
            camarilla_h4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
            if not np.isnan(camarilla_h4_1d_aligned[i]) and close[i] >= camarilla_h4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_12hEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0