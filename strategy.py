#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Use 12h timeframe with Camarilla R3/S3 breakout, confirmed by 1w EMA34 trend and volume spike.
Long when: price breaks above R3 + 1w EMA34 uptrend + volume > 1.8 * avg volume.
Short when: price breaks below S3 + 1w EMA34 downtrend + volume > 1.8 * avg volume.
Exit when: price reverts to Camarilla midpoint (PP) or touches opposite level (S2/R2).
Target: 12-37 trades/year to avoid fee drag. Works in bull/bear via 1w trend filter.
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
    
    # Calculate Camarilla levels from previous week (using 1w HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for Camarilla calculation
    prev_high = df_1w['high'].shift(1).values  # shift(1) for previous week
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Camarilla levels: R3, S3, PP (pivot point), R2, S2 for exit
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_r2 = prev_close + (prev_high - prev_low) * 1.1 / 6
    camarilla_s2 = prev_close - (prev_high - prev_low) * 1.1 / 6
    
    # Align to 12h timeframe (wait for completed 1w bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pp)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s2)
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: current volume > 1.8 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for volume avg, 34 for 1w EMA
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r2_aligned[i]) or
            np.isnan(camarilla_s2_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above R3 + 1w EMA34 uptrend + volume spike
            long_entry = (close_val > camarilla_r3_aligned[i]) and \
                       (ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]) and \
                       volume_spike[i]
            # Short: break below S3 + 1w EMA34 downtrend + volume spike
            short_entry = (close_val < camarilla_s3_aligned[i]) and \
                        (ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]) and \
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
            # Long - exit when price reverts to PP or touches S2 (contrarian exit)
            if (close_val < camarilla_pp_aligned[i]) or (close_val < camarilla_s2_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price reverts to PP or touches R2 (contrarian exit)
            if (close_val > camarilla_pp_aligned[i]) or (close_val > camarilla_r2_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0