#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA34_Trend_VolumeSpike_v1
Hypothesis: Trade 4h Camarilla R1/S1 breakouts filtered by 12h EMA34 trend and volume spike.
Camarilla pivot levels provide high-probability reversal/breakout points. 12h EMA34 filters for trend alignment,
reducing false breakouts. Volume spike confirms institutional participation. Designed for 4h timeframe
to target 25-50 trades/year, minimizing fee drag while capturing significant moves in both bull and bear markets.
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels: R1, S1, R2, S2, R3, S3, R4, S4
    # Camarilla: Close = previous day close, High = previous day high, Low = previous day low
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate pivot and ranges
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels (based on previous day's data)
    R1_1d = close_1d + (range_1d * 1.1 / 12)
    S1_1d = close_1d - (range_1d * 1.1 / 12)
    R2_1d = close_1d + (range_1d * 1.1 / 6)
    S2_1d = close_1d - (range_1d * 1.1 / 6)
    R3_1d = close_1d + (range_1d * 1.1 / 4)
    S3_1d = close_1d - (range_1d * 1.1 / 4)
    R4_1d = close_1d + (range_1d * 1.1 / 2)
    S4_1d = close_1d - (range_1d * 1.1 / 2)
    
    # 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: 2.0x median volume over 30 periods
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    # Align HTF indicators to 4h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    R2_1d_aligned = align_htf_to_ltf(prices, df_1d, R2_1d)
    S2_1d_aligned = align_htf_to_ltf(prices, df_1d, S2_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Warmup: max of daily pivot (2), 12h EMA (34), volume median (30)
    start_idx = max(2, 34, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_median[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        r1_val = R1_1d_aligned[i]
        s1_val = S1_1d_aligned[i]
        r2_val = R2_1d_aligned[i]
        s2_val = S2_1d_aligned[i]
        r3_val = R3_1d_aligned[i]
        s3_val = S3_1d_aligned[i]
        r4_val = R4_1d_aligned[i]
        s4_val = S4_1d_aligned[i]
        ema_34_12h_val = ema_34_12h_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume and trend alignment
            long_signal = (close_val > r1_val) and \
                          (close_val > ema_34_12h_val) and \
                          (volume_val > 2.0 * vol_median_val)
            # Short breakdown: price breaks below S1 with volume and trend alignment
            short_signal = (close_val < s1_val) and \
                           (close_val < ema_34_12h_val) and \
                           (volume_val > 2.0 * vol_median_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long with minimum holding period
            bars_since_entry += 1
            signals[i] = 0.25
            # Exit: price breaks below S1 (reversal) or trend changes after minimum holding
            if bars_since_entry >= 6 and ((close_val < s1_val) or (close_val < ema_34_12h_val)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short with minimum holding period
            bars_since_entry += 1
            signals[i] = -0.25
            # Exit: price breaks above R1 (reversal) or trend changes after minimum holding
            if bars_since_entry >= 6 and ((close_val > r1_val) or (close_val > ema_34_12h_val)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0