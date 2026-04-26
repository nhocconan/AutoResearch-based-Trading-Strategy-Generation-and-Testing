#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike on 6h timeframe. 
Camarilla levels provide institutional support/resistance. Breakouts above R3 or below S3 with 
trend alignment and volume confirmation capture strong moves. Target 12-37 trades/year to minimize 
fee drag. Works in bull/bear via trend filter.
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
    
    # Calculate 1d OHLC for Camarilla pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Daily pivot: (daily_high + daily_low + daily_close) / 3
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_range = daily_high - daily_low
    
    # Camarilla levels: R3 = pivot + 1.1 * range / 2, S3 = pivot - 1.1 * range / 2
    camarilla_r3 = daily_pivot + (1.1 * daily_range / 2.0)
    camarilla_s3 = daily_pivot - (1.1 * daily_range / 2.0)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25  # 25% position size
    
    # Warmup: need enough for all indicators
    start_idx = max(20, 34)  # volume avg, EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above Camarilla R3 + price above EMA34 + volume spike
            long_entry = (close_val > camarilla_r3_aligned[i]) and \
                       (close_val > ema_34_aligned[i]) and \
                       volume_spike[i]
            # Short: break below Camarilla S3 + price below EMA34 + volume spike
            short_entry = (close_val < camarilla_s3_aligned[i]) and \
                        (close_val < ema_34_aligned[i]) and \
                        volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on close below EMA34 or Camarilla S3 (mean reversion)
            exit_condition = (close_val < ema_34_aligned[i]) or \
                           (close_val < camarilla_s3_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on close above EMA34 or Camarilla R3 (mean reversion)
            exit_condition = (close_val > ema_34_aligned[i]) or \
                           (close_val > camarilla_r3_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0