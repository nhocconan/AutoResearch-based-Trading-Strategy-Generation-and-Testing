#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: On 4h timeframe, price touching Camarilla R1/S1 levels with 1d trend filter (EMA34) and volume confirmation (>2x avg) provides high-probability mean-reversion entries in ranging markets and breakout entries in trending markets. Long when price touches S1 in uptrend with volume, short when touches R1 in downtrend with volume. Uses discrete sizing (0.0, ±0.25) to minimize fee churn. Targets 75-200 trades over 4 years (19-50/year) for optimal 4h frequency. Daily trend filter ensures alignment with higher timeframe momentum while volume spike confirms institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need enough for EMA34
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day's range
    # Need at least 1 day of data
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d_series = pd.Series(df_1d['close'].values)
    
    # Previous day's range
    prev_high = high_1d.shift(1).values
    prev_low = low_1d.shift(1).values
    prev_close = close_1d_series.shift(1).values
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    camarilla_r1 = prev_close + (range_ * 1.1 / 12)
    camarilla_s1 = prev_close - (range_ * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume ratio (current / 20-period average) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup + volume MA + Camarilla data
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_ratio[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        vol_confirmed = vol_ratio[i] > 2.0  # volume at least 2.0x average
        
        if position == 0:
            # Long: price touches S1 in uptrend (price > EMA34) with volume
            long_signal = (close[i] <= camarilla_s1_aligned[i] * 1.001 and  # allow small tolerance
                          close[i] >= camarilla_s1_aligned[i] * 0.999 and
                          close[i] > ema_34_1d_aligned[i] and
                          vol_confirmed)
            
            # Short: price touches R1 in downtrend (price < EMA34) with volume
            short_signal = (close[i] >= camarilla_r1_aligned[i] * 0.999 and  # allow small tolerance
                           close[i] <= camarilla_r1_aligned[i] * 1.001 and
                           close[i] < ema_34_1d_aligned[i] and
                           vol_confirmed)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below EMA34 OR reaches Camarilla R3 (take profit)
            camarilla_r3 = camarilla_r1_aligned[i] * 2  # approximate R3 as 2x R1 distance from close
            if close[i] < ema_34_1d_aligned[i] or close[i] >= camarilla_r3:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above EMA34 OR reaches Camarilla S3 (take profit)
            camarilla_s3 = camarilla_s1_aligned[i] * 2  # approximate S3 as 2x S1 distance from close
            if close[i] > ema_34_1d_aligned[i] or close[i] <= camarilla_s3:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0