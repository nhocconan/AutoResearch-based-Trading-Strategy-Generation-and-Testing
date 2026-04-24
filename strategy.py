#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike.
- Primary timeframe: 12h, HTF: 1d for EMA34 trend alignment.
- Camarilla pivot levels from prior 1d: long at H3 breakout, short at L3 breakdown.
- Trend filter: only long when 12h close > 1d EMA34, only short when 12h close < 1d EMA34.
- Volume confirmation: current 12h volume > 2.0 * 20-period 12h volume MA (strict filter).
- Discrete signal size: 0.25 to minimize fee churn and control drawdown.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in both bull/bear: trend filter avoids counter-trend trades, Camarilla levels act as dynamic support/resistance.
- Exit: price reverts to Camarilla pivot point (PP) from prior 1d.
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d (use completed 1d bar)
    # PP = (H + L + C) / 3
    # H3 = PP + (H - L) * 1.1 / 2
    # L3 = PP - (H - L) * 1.1 / 2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    pp = (high_1d + low_1d + close_1d_arr) / 3.0
    h3 = pp + (high_1d - low_1d) * 1.1 / 2.0
    l3 = pp - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe (completed 1d bar only)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Trend filter: 12h close vs 1d EMA34
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 34, 20)  # Need 1d EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above H3 AND uptrend AND volume spike
            if close[i] > h3_aligned[i] and uptrend[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below L3 AND downtrend AND volume spike
            elif close[i] < l3_aligned[i] and downtrend[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to pivot point (PP) or reverse signal
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to pivot point (PP) or reverse signal
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3_L3_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0