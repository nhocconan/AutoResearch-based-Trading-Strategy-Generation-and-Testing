#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and 1d volume spike filter.
- Primary timeframe: 1h, HTF: 4h for EMA50 trend alignment and 1d for volume confirmation.
- Camarilla pivot levels from prior 1d: long at H3 breakout, short at L3 breakdown.
- Trend filter: only long when 1h close > 4h EMA50, only short when 1h close < 4h EMA50.
- Volume confirmation: current 1h volume > 2.0 * 20-period 1d volume MA (strict filter).
- Discrete signal size: 0.20 to minimize fee churn and control drawdown.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
- Exit: price reverts to Camarilla pivot point (PP) from prior 1d.
- Session filter: only trade between 08:00-20:00 UTC to reduce noise.
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
    
    # Precompute session hours (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from prior 1d (use completed 1d bar)
    # PP = (H + L + C) / 3
    # H3 = PP + (H - L) * 1.1 / 2
    # L3 = PP - (H - L) * 1.1 / 2
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    pp = (high_1d + low_1d + close_1d_arr) / 3.0
    h3 = pp + (high_1d - low_1d) * 1.1 / 2.0
    l3 = pp - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 1h timeframe (completed 1d bar only)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Volume confirmation: current 1h volume > 2.0 * 20-period 1d volume MA
    # We need 1d volume data to compute the MA
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    # Trend filter: 1h close vs 4h EMA50
    uptrend = close > ema_50_4h_aligned
    downtrend = close < ema_50_4h_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20)  # Need 4h EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i] or \
           np.isnan(ema_50_4h_aligned[i]) or np.isnan(h3_aligned[i]) or \
           np.isnan(l3_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above H3 AND uptrend AND volume spike
            if close[i] > h3_aligned[i] and uptrend[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: price closes below L3 AND downtrend AND volume spike
            elif close[i] < l3_aligned[i] and downtrend[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price reverts to pivot point (PP) or reverse signal
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price reverts to pivot point (PP) or reverse signal
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA50_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0