#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike.
- Primary timeframe: 12h, HTF: 1d for trend filter and Camarilla calculation
- Long: Close breaks above R3 + price > 1d EMA34 (uptrend) + volume > 2.0x 20-period avg
- Short: Close breaks below S3 + price < 1d EMA34 (downtrend) + volume > 2.0x 20-period avg
- Exit: Close reverts to pivot point (PP) of Camarilla levels
- Uses wider Camarilla breakouts (R3/S3) for fewer, higher-quality entries on 12h
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.30 to balance return and risk
- BTC/ETH focus: requires HTF trend alignment to avoid SOL-only bias
- Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)
- Uses mtf_data helper for proper HTF alignment without look-ahead
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
    
    # Volume confirmation: > 2.0x 20-period average (volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous day using 1d data
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
    range_1d = high_1d - low_1d
    r3 = close_1d + 1.1 * range_1d / 4.0
    s3 = close_1d - 1.1 * range_1d / 4.0
    pp = (high_1d + low_1d + close_1d) / 3.0  # Pivot point
    
    # Align to 12h timeframe (values from previous 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # Need 20 for volume MA, 34 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above R3 + price > 1d EMA34 (uptrend) + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike):
                signals[i] = 0.30
                position = 1
            # Short: Close breaks below S3 + price < 1d EMA34 (downtrend) + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: Close reverts to pivot point (PP)
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: Close reverts to pivot point (PP)
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0