#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: 12h strategy using Camarilla R3/S3 breakouts with 1w EMA50 trend filter and volume confirmation. 
Uses weekly trend to capture major market direction (works in both bull/bear via trend alignment). 
R3/S3 levels from prior weekly bar reduce false breakouts. Volume spike confirms institutional participation. 
Designed for BTC/ETH robustness with tight entry conditions (~12-37 trades/year). 
Position size 0.25 to manage drawdown. Uses discrete levels to minimize fee drag.
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Get 1w data for Camarilla R3/S3 levels (from previous completed 1w bar)
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    rng = prev_high - prev_low
    r3 = prev_close + (rng * 1.50)   # R3 level
    s3 = prev_close - (rng * 1.50)   # S3 level
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Fixed position size to minimize churn
    
    # Warmup: need 1w EMA50 (50), 1w shift(1) for Camarilla, vol avg (20)
    start_idx = max(50 + 1, 1 + 1, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Camarilla R3/S3 breakout with 1w EMA50 alignment and volume confirmation
            long_condition = (close_val > r3_val and 
                            close_val > ema_val and 
                            vol_conf)
            short_condition = (close_val < s3_val and 
                             close_val < ema_val and 
                             vol_conf)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1w EMA50 (trend reversal)
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 1w EMA50 (trend reversal)
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0