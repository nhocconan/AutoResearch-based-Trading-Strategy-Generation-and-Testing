#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike
Hypothesis: On 1h timeframe, trade Camarilla R3/S3 breakouts with 4h EMA50 trend filter and 1d volume spike confirmation.
Long when: price breaks above R3 + 4h EMA50 uptrend + 1d volume > 1.8 * 20-period average.
Short when: price breaks below S3 + 4h EMA50 downtrend + 1d volume > 1.8 * 20-period average.
Exit when: price reverts to Camarilla midpoint (PP) or touches opposite Camarilla level (S3 for longs, R3 for shorts).
Session filter: 08:00-20:00 UTC to avoid low-liquidity hours.
Position size: 0.20 discrete to minimize fee churn.
Target: 15-35 trades/year (~60-140 total over 4 years) to avoid fee drag.
Uses 4h/1d for signal direction, 1h only for entry timing precision.
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
    
    # Precompute session hours filter (08:00-20:00 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Camarilla levels from previous day (using 1d HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # shift(1) for previous day
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R3, S3, PP (pivot point)
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    
    # Align to 1h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d volume spike: current 1d volume > 1.8 * 20-period average
    df_1d_vol = get_htf_data(prices, '1d')
    if len(df_1d_vol) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d_vol['volume'].values
    vol_avg_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (1.8 * vol_avg_20)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d_vol, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for volume avg, 50 for 4h EMA
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i] or \
           np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(camarilla_pp_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or \
           np.isnan(volume_spike_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.20  # Fixed position size
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above R3 + 4h EMA50 uptrend + 1d volume spike
            long_entry = (close_val > camarilla_r3_aligned[i]) and \
                       (ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]) and \
                       volume_spike_1d_aligned[i]
            # Short: break below S3 + 4h EMA50 downtrend + 1d volume spike
            short_entry = (close_val < camarilla_s3_aligned[i]) and \
                        (ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]) and \
                        volume_spike_1d_aligned[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price reverts to PP or touches S3 (contrarian exit)
            if (close_val < camarilla_pp_aligned[i]) or (close_val < camarilla_s3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price reverts to PP or touches R3 (contrarian exit)
            if (close_val > camarilla_pp_aligned[i]) or (close_val > camarilla_r3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0