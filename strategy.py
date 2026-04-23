#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike.
- Primary timeframe: 1h, HTF: 4h for trend filter, 1d for session context
- Long: Close breaks above R3 + price > 4h EMA50 (uptrend) + volume > 2.0x 24-period avg
- Short: Close breaks below S3 + price < 4h EMA50 (downtrend) + volume > 2.0x 24-period avg
- Exit: Close reverts to pivot point (PP) of Camarilla levels
- Session filter: 08:00-20:00 UTC only (avoid low-liquidity hours)
- Uses tighter Camarilla breakouts (R3/S3) for fewer, higher-quality entries
- Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe
- Discrete position sizing: ±0.20 to minimize fee churn
- Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Volume confirmation: > 2.0x 24-period average (volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate Camarilla levels from previous 1h bar
    # Need 1h high, low, close - use 1h data shifted by 1
    df_1h = get_htf_data(prices, '1h')
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Camarilla levels: based on previous 1h bar's range
    # R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
    range_1h = high_1h - low_1h
    r3 = close_1h + 1.1 * range_1h / 4.0
    s3 = close_1h - 1.1 * range_1h / 4.0
    pp = (high_1h + low_1h + close_1h) / 3.0  # Pivot point
    
    # Align to 1h timeframe (values from previous 1h bar)
    r3_aligned = align_htf_to_ltf(prices, df_1h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1h, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1h, pp)
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 1)  # Need 24 for volume MA, 1 for Camarilla (aligned from 1h)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(vol_ma[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above R3 + price > 4h EMA50 (uptrend) + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike):
                signals[i] = 0.20
                position = 1
            # Short: Close breaks below S3 + price < 4h EMA50 (downtrend) + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Close reverts to pivot point (PP)
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Close reverts to pivot point (PP)
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike"
timeframe = "1h"
leverage = 1.0