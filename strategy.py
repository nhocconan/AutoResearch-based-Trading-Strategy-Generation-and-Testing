#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla H3/L3 breakout with 1w EMA34 trend filter and volume spike.
- Primary timeframe: 1d, HTF: 1w for EMA34 trend alignment.
- Camarilla pivot levels from prior 1w: long at H3 breakout, short at L3 breakdown.
- Trend filter: only long when 1d close > 1w EMA34, only short when 1d close < 1w EMA34.
- Volume confirmation: current 1d volume > 2.0 * 20-period 1d volume MA (strict filter).
- Discrete signal size: 0.25 to minimize fee churn and control drawdown.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
- Exit: price reverts to Camarilla pivot point (PP) from prior 1w.
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
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from prior 1w (use completed 1w bar)
    # PP = (H + L + C) / 3
    # H3 = PP + (H - L) * 1.1 / 2
    # L3 = PP - (H - L) * 1.1 / 2
    df_1w_prior = get_htf_data(prices, '1w')  # Same data, will be aligned with delay
    if len(df_1w_prior) < 1:
        return np.zeros(n)
    
    high_1w = df_1w_prior['high'].values
    low_1w = df_1w_prior['low'].values
    close_1w_arr = df_1w_prior['close'].values
    
    pp = (high_1w + low_1w + close_1w_arr) / 3.0
    h3 = pp + (high_1w - low_1w) * 1.1 / 2.0
    l3 = pp - (high_1w - low_1w) * 1.1 / 2.0
    
    # Align Camarilla levels to 1d timeframe (completed 1w bar only)
    h3_aligned = align_htf_to_ltf(prices, df_1w_prior, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w_prior, l3)
    pp_aligned = align_htf_to_ltf(prices, df_1w_prior, pp)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Trend filter: 1d close vs 1w EMA34
    uptrend = close > ema_34_1w_aligned
    downtrend = close < ema_34_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 34, 20)  # Need 1w EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(h3_aligned[i]) or 
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

name = "1d_Camarilla_H3L3_1wEMA34_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0