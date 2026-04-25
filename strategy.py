#!/usr/bin/env python3
"""
1h_Camarilla_R3S3_Breakout_4hTrend_1dVolSpike
Hypothesis: Trade 1h Camarilla R3/S3 breakouts with 4h EMA50 trend filter and 1d volume spike confirmation.
Use 4h for signal direction (trend filter), 1d for regime filter (volume spike), and 1h only for entry timing precision.
Target: 60-150 total trades over 4 years = 15-37/year to stay within fee drag limits.
Works in bull/bear: 4h trend filter avoids counter-trend whipsaws, volume spike confirms institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume spike regime filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d volume MA20 for spike detection
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_20_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Calculate Camarilla levels from previous day's OHLC (using 1d data)
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    camarilla_range = prev_day_high - prev_day_low
    r3 = prev_day_close + 1.1 * camarilla_range / 4  # R3 level
    s3 = prev_day_close - 1.1 * camarilla_range / 4  # S3 level
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 4h EMA50 (50), 1d volume MA (20), and Camarilla (1)
    start_idx = max(50, 20, 1)
    
    # Precompute session filter (08-20 UTC) - assuming open_time is datetime64
    hours = prices.index.hour if isinstance(prices.index, pd.DatetimeIndex) else pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC only
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND 4h trend bullish (close > EMA50) AND 1d volume spike
            long_setup = (close[i] > r3_aligned[i]) and \
                         (close[i] > ema_50_4h_aligned[i]) and \
                         (volume_spike_1d_aligned[i] > 0.5)
            # Short: price breaks below S3 AND 4h trend bearish (close < EMA50) AND 1d volume spike
            short_setup = (close[i] < s3_aligned[i]) and \
                          (close[i] < ema_50_4h_aligned[i]) and \
                          (volume_spike_1d_aligned[i] > 0.5)
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: price re-enters Camarilla range OR 4h trend turns bearish
            if (close[i] < r3_aligned[i] and close[i] > s3_aligned[i]) or \
               (close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price re-enters Camarilla range OR 4h trend turns bullish
            if (close[i] < r3_aligned[i] and close[i] > s3_aligned[i]) or \
               (close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hTrend_1dVolSpike"
timeframe = "1h"
leverage = 1.0