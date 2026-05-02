#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d ADX trend filter and volume spike
# Uses Camarilla pivot levels from 1d for structure, 1d ADX(14) > 25 for trending regime
# Volume spike (1.8x 20-period average) ensures participation
# Only takes breakouts in direction of 1d trend to avoid counter-trend whipsaws
# Discrete position sizing 0.25 minimizes fee churn
# Targets 12-30 trades/year (50-120 total over 4 years) to stay within fee drag limits
# ADX filter reduces false signals in ranging markets, improving Sharpe in bear/range regimes

name = "4h_Camarilla_R3S3_1dADX_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivot and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3 = pivot + range_1d * 1.1 / 2.0
    s3 = pivot - range_1d * 1.1 / 2.0
    
    # Calculate 1d ADX(14) for trend filter
    # ADX calculation: +DI, -DI, DX, then smoothed ADX
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = -np.diff(low_1d, prepend=low_1d[0])  # negative of low diff
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_smooth = wilders_smoothing(tr, period)
    plus_dm_smooth = wilders_smoothing(plus_dm, period)
    minus_dm_smooth = wilders_smoothing(minus_dm, period)
    
    # Avoid division by zero
    plus_di = np.where(tr_smooth != 0, (plus_dm_smooth / tr_smooth) * 100, 0)
    minus_di = np.where(tr_smooth != 0, (minus_dm_smooth / tr_smooth) * 100, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 
                  np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100, 0)
    
    # Wilder's smoothing of DX to get ADX
    adx = wilders_smoothing(dx, period)
    
    # Align Camarilla levels and ADX to 4h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 4h volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for ADX and volume MA)
    start_idx = 54  # max(34 for ADX smoothing, 20 for volume) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend strength from 1d ADX (> 25 = trending)
        trending = adx_aligned[i] > 25.0
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3 AND trending AND volume confirm
            if (close[i] > r3_aligned[i] and 
                trending and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND trending AND volume confirm
            elif (close[i] < s3_aligned[i] and 
                  trending and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below S3 OR trend weakens (ADX < 20)
            if (close[i] < s3_aligned[i] or 
                adx_aligned[i] < 20.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above R3 OR trend weakens (ADX < 20)
            if (close[i] > r3_aligned[i] or 
                adx_aligned[i] < 20.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals