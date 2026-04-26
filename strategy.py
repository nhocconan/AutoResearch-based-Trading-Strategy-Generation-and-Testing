#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above R3 AND close > 1d EMA34 AND volume > 1.5x 20-period avg volume.
Short when price breaks below S3 AND close < 1d EMA34 AND volume > 1.5x 20-period avg volume.
Exit on opposite Camarilla level touch (R1 for longs, S1 for shorts) or trend reversal.
Uses discrete sizing (0.25) to minimize fee drag. Target: 75-200 trades over 4 years.
Works in bull/bear via 1d trend filter and volatility-based Camarilla levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Camarilla levels (based on previous day's range)
    # Calculate daily pivot points from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # S4 = close - 1.5*(high-low), S3 = close - 1.1*(high-low), etc.
    # R3/S3 are the primary breakout levels
    # R1/S1 are used for exits
    
    # Previous day's values (shifted by 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    # First bar has no previous day
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    # Calculate Camarilla levels for 1d
    camarilla_range = prev_high_1d - prev_low_1d
    r3_1d = prev_close_1d + 1.1 * camarilla_range
    s3_1d = prev_close_1d - 1.1 * camarilla_range
    r1_1d = prev_close_1d + 0.5 * camarilla_range
    s1_1d = prev_close_1d - 0.5 * camarilla_range
    
    # Align 1d Camarilla levels to 4h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 34-period for EMA and valid Camarilla)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long conditions: price breaks above R3, above 1d EMA34, volume spike
        long_breakout = close[i] > r3_1d_aligned[i]
        long_trend = close[i] > ema_34_1d_aligned[i]
        long_volume = volume_spike[i]
        
        # Short conditions: price breaks below S3, below 1d EMA34, volume spike
        short_breakout = close[i] < s3_1d_aligned[i]
        short_trend = close[i] < ema_34_1d_aligned[i]
        short_volume = volume_spike[i]
        
        # Exit conditions
        long_exit = close[i] < r1_1d_aligned[i]  # Exit long at R1
        short_exit = close[i] > s1_1d_aligned[i]  # Exit short at S1
        trend_reversal_long = close[i] < ema_34_1d_aligned[i]  # Exit long if trend turns bearish
        trend_reversal_short = close[i] > ema_34_1d_aligned[i]  # Exit short if trend turns bullish
        
        # Entry logic
        if long_breakout and long_trend and long_volume:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        elif short_breakout and short_trend and short_volume:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit logic
        elif position == 1 and (long_exit or trend_reversal_long):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (short_exit or trend_reversal_short):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0