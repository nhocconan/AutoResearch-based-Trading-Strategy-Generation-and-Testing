#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: On 6h timeframe, Camarilla R3/S3 breakouts with weekly trend filter (price > weekly EMA50 for longs, < for shorts) and volume confirmation capture strong momentum continuations in both bull and bear markets. Weekly EMA50 adapts to longer-term trend, reducing whipsaws. Discrete position sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits for 6h.
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
    
    # Get 1d data for Camarilla levels (from previous completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    r3 = prev_close + (rng * 1.1 / 4)  # R3 = C + 1.1*(H-L)/4
    s3 = prev_close - (rng * 1.1 / 4)  # S3 = C - 1.1*(H-L)/4
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Get 1w data for weekly EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike: current > 2.0 * 20-period average (strict to reduce trade frequency)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # Discrete size to reduce fee churn
    
    # Warmup: need 1d shift, 1w EMA50, vol avg
    start_idx = max(30, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_w_val = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry: Camarilla R3/S3 breakout with weekly trend alignment and volume spike
            long_condition = (close_val > r3_val and 
                            close_val > ema_w_val and 
                            vol_spike)
            short_condition = (close_val < s3_val and 
                             close_val < ema_w_val and 
                             vol_spike)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: price crosses below weekly EMA50 (trend reversal)
            if close_val < ema_w_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above weekly EMA50 (trend reversal)
            if close_val > ema_w_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0