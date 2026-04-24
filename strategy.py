#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and ADX trend filter.
- Long when price breaks above Camarilla R3 AND 1d volume > 2.0 * 20-period average AND 1d ADX > 25
- Short when price breaks below Camarilla S3 AND 1d volume > 2.0 * 20-period average AND 1d ADX > 25
- Exit when price retreats to Camarilla H3/L3 levels OR ADX < 20 (trend weakness)
- Uses Camarilla levels from 1d for structure, 4h for execution, volume spike for conviction, ADX for trend filter
- Discrete signal size: 0.30 to balance capture and drawdown control
- Target: 80-180 total trades over 4 years (20-45/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # R3 = Close + 1.1 * (High - Low) / 2
    # S3 = Close - 1.1 * (High - Low) / 2
    # H3 = Close + 1.1 * (High - Low) / 4
    # L3 = Close - 1.1 * (High - Low) / 4
    rng = prev_high - prev_low
    camarilla_r3 = prev_close + 1.1 * rng / 2
    camarilla_s3 = prev_close - 1.1 * rng / 2
    camarilla_h3 = prev_close + 1.1 * rng / 4
    camarilla_l3 = prev_close - 1.1 * rng / 4
    
    # Align 1d Camarilla levels to 4h (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d ADX for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Wilder's smoothing
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        if len(values) < period:
            return result
        result[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    period = 14
    tr_smoothed = wilders_smoothing(tr, period)
    dm_plus_smoothed = wilders_smoothing(dm_plus, period)
    dm_minus_smoothed = wilders_smoothing(dm_minus, period)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smoothed / tr_smoothed
    di_minus = 100 * dm_minus_smoothed / tr_smoothed
    di_plus = np.where(tr_smoothed == 0, 0, di_plus)
    di_minus = np.where(tr_smoothed == 0, 0, di_minus)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx_1d = wilders_smoothing(dx, period)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: 1d volume > 2.0 * 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Trend filter: trending if ADX > 25, weak if ADX < 20
    strong_trend = adx_1d_aligned > 25
    weak_trend = adx_1d_aligned < 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where indicators are ready
    start_idx = max(20, 30) + 1  # Need volume MA and ADX data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > R3 AND volume spike AND strong trend
            if close[i] > camarilla_r3_aligned[i] and volume_spike_1d_aligned[i] and strong_trend[i]:
                signals[i] = 0.30
                position = 1
            # Short: price < S3 AND volume spike AND strong trend
            elif close[i] < camarilla_s3_aligned[i] and volume_spike_1d_aligned[i] and strong_trend[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price < H3 OR weak trend
            if close[i] < camarilla_h3_aligned[i] or weak_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price > L3 OR weak trend
            if close[i] > camarilla_l3_aligned[i] or weak_trend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dVolumeSpike_ADXTrend_v1"
timeframe = "4h"
leverage = 1.0