#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirmed_v1
Hypothesis: On 1h timeframe, Camarilla R1/S1 breakouts with 4h EMA50 trend filter and volume confirmation produce high-probability trades. The 4h EMA50 establishes the primary trend, while Camarilla levels provide precise entry/exit points. Volume confirmation reduces false breakouts. Uses session filter (08-20 UTC) to avoid low-liquidity periods. Target: 60-150 total trades over 4 years (15-37/year) with strict entry conditions to minimize fee drag.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # Load 4h data ONCE before loop for HTF trend filter (EMA50) and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h Camarilla levels (R1, S1, R3, S3)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_shifted = np.concatenate([[np.nan], close_4h[:-1]])  # previous 4h bar close
    
    # True range for previous 4h bar
    tr_4h = np.maximum(
        high_4h[1:] - low_4h[1:],
        np.maximum(
            np.abs(high_4h[1:] - close_4h_shifted[1:]),
            np.abs(low_4h[1:] - close_4h_shifted[1:])
        )
    )
    tr_4h = np.concatenate([[np.nan], tr_4h])
    
    # Camarilla calculation uses previous 4h bar's OHLC
    camarilla_range = high_4h - low_4h
    r1 = close_4h_shifted + 1.1 * camarilla_range / 12
    s1 = close_4h_shifted - 1.1 * camarilla_range / 12
    r3 = close_4h_shifted + 1.1 * camarilla_range / 4
    s3 = close_4h_shifted - 1.1 * camarilla_range / 4
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # 1h volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # 4h trend filter (EMA50)
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        # Volume confirmation
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Camarilla breakout conditions
        breakout_r1 = close[i] > r1_aligned[i]
        breakout_s1 = close[i] < s1_aligned[i]
        breakout_r3 = close[i] > r3_aligned[i]
        breakout_s3 = close[i] < s3_aligned[i]
        
        # Long logic: breakout above R1 in uptrend with volume
        if uptrend and volume_spike and breakout_r1:
            if position != 1:
                signals[i] = 0.20
                position = 1
            else:
                signals[i] = 0.20
        # Short logic: breakout below S1 in downtrend with volume
        elif downtrend and volume_spike and breakout_s1:
            if position != -1:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = -0.20
        # Exit conditions: breakout beyond R3/S3 or loss of trend
        elif position == 1 and (breakout_r3 or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (breakout_s3 or not downtrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirmed_v1"
timeframe = "1h"
leverage = 1.0