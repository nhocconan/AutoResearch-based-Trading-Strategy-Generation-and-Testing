#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Trade Camarilla pivot (R3/S3) breakouts on 6h with 1d EMA50 trend filter and volume spike confirmation.
Uses R3/S3 levels for stronger breakout signals (less noise than R1/S1) and 1d EMA50 for smoother trend.
Volume confirmation requires 2.0x 20-period median volume to reduce false breakouts.
Only trade in trending markets (ADX > 20) to avoid whipsaws in ranging markets.
Designed for 12-25 trades/year on BTC/ETH/SOL with controlled risk.
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
    
    # Get 1d data for HTF trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter (smoother than EMA34)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous 1d OHLC
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    camarilla_r3 = prev_close_1d + 1.500 * (prev_high_1d - prev_low_1d)
    camarilla_s3 = prev_close_1d - 1.500 * (prev_high_1d - prev_low_1d)
    
    # Align HTF indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 2.0x median volume (20-period)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # ADX(14) for regime filter - trending when > 20
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(high[i] - high[i-1], 0) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(low[i-1] - low[i], 0) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Wilder's smoothing
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    if n >= period:
        plus_dm_smooth = WilderSmooth(plus_dm, period)
        minus_dm_smooth = WilderSmooth(minus_dm, period)
        tr_smooth = WilderSmooth(tr, period)
        
        # Avoid division by zero
        plus_di = 100 * plus_dm_smooth / np.where(tr_smooth != 0, tr_smooth, 1)
        minus_di = 100 * minus_dm_smooth / np.where(tr_smooth != 0, tr_smooth, 1)
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), 1)
        adx = WilderSmooth(dx, period)
    else:
        adx = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 1d EMA (50), volume median (20), ADX (14*2 for smoothing)
    start_idx = max(50, 20, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_median[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(adx[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        adx_val = adx[i]
        
        if position == 0:
            # Long: break above R3 with volume spike, uptrend, and trending regime
            long_signal = (close_val > camarilla_r3_aligned[i]) and \
                          (volume_val > 2.0 * vol_median_val) and \
                          (close_val > ema_50_1d_val) and \
                          (adx_val > 20)
            
            # Short: break below S3 with volume spike, downtrend, and trending regime
            short_signal = (close_val < camarilla_s3_aligned[i]) and \
                           (volume_val > 2.0 * vol_median_val) and \
                           (close_val < ema_50_1d_val) and \
                           (adx_val > 20)
            
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
            # Exit: price breaks below S3 (reversal) or trend changes (close < 1d EMA50) or regime changes (ADX < 15)
            if (close_val < camarilla_s3_aligned[i]) or \
               (close_val < ema_50_1d_val) or \
               (adx_val < 15):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R3 (reversal) or trend changes (close > 1d EMA50) or regime changes (ADX < 15)
            if (close_val > camarilla_r3_aligned[i]) or \
               (close_val > ema_50_1d_val) or \
               (adx_val < 15):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0