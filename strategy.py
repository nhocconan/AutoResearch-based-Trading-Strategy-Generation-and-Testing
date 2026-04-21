#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Fade_VolumeSpike_v1
Hypothesis: Fade extreme Camarilla levels (R3/S3) on 6h with 1d volume spike confirmation and 1d EMA50 trend filter.
Works in bull/bear: In uptrend, fade R3 for short; in downtrend, fade S3 for long. Volume spike confirms exhaustion.
Target: 15-30 trades/year per symbol (60-120 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels, trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R3, S3 (extreme fade levels)
    rang = prev_high - prev_low
    r3 = prev_close + rang * 3.0 / 12
    s3 = prev_close - rang * 3.0 / 12
    
    # Align to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume spike: current volume > 2.0 * 20-day average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (2.0 * vol_ma_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        if position == 0:
            # Long conditions: price < S3 (oversold) AND 1d uptrend AND volume spike
            if (price < s3_aligned[i] and 
                ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and  # 1d EMA rising
                vol_spike_aligned[i] > 0.5):  # volume spike confirmed
                signals[i] = 0.25
                position = 1
            # Short conditions: price > R3 (overbought) AND 1d downtrend AND volume spike
            elif (price > r3_aligned[i] and 
                  ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and  # 1d EMA falling
                  vol_spike_aligned[i] > 0.5):  # volume spike confirmed
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price > 1d EMA50 (trend exhaustion) or price > R2 (mean reversion fail)
            prev_high_i = high_1d[np.searchsorted(df_1d.index, prices['open_time'].iloc[i]) - 1] if i > 0 else np.nan
            prev_low_i = low_1d[np.searchsorted(df_1d.index, prices['open_time'].iloc[i]) - 1] if i > 0 else np.nan
            prev_close_i = close_1d[np.searchsorted(df_1d.index, prices['open_time'].iloc[i]) - 1] if i > 0 else np.nan
            if not (np.isnan(prev_high_i) or np.isnan(prev_low_i) or np.isnan(prev_close_i)):
                rang_i = prev_high_i - prev_low_i
                r2_exit = prev_close_i + rang_i * 2.0 / 12
                # Simple approach: use current day's R2 level (already calculated)
                r2_current = prev_close + rang * 2.0 / 12
                r2_aligned = align_htf_to_ltf(prices, df_1d, r2_current)
                if price > ema_50_1d_aligned[i] or (not np.isnan(r2_aligned[i]) and price > r2_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price < 1d EMA50 (trend exhaustion) or price < S2 (mean reversion fail)
            prev_high_i = high_1d[np.searchsorted(df_1d.index, prices['open_time'].iloc[i]) - 1] if i > 0 else np.nan
            prev_low_i = low_1d[np.searchsorted(df_1d.index, prices['open_time'].iloc[i]) - 1] if i > 0 else np.nan
            prev_close_i = close_1d[np.searchsorted(df_1d.index, prices['open_time'].iloc[i]) - 1] if i > 0 else np.nan
            if not (np.isnan(prev_high_i) or np.isnan(prev_low_i) or np.isnan(prev_close_i)):
                rang_i = prev_high_i - prev_low_i
                s2_exit = prev_close_i - rang_i * 2.0 / 12
                # Simple approach: use current day's S2 level
                s2_current = prev_close - rang * 2.0 / 12
                s2_aligned = align_htf_to_ltf(prices, df_1d, s2_current)
                if price < ema_50_1d_aligned[i] or (not np.isnan(s2_aligned[i]) and price < s2_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Fade_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0