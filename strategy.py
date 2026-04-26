#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_SessionFilter
Hypothesis: Use 1h timeframe with Camarilla R1/S1 breakout from prior day, confirmed by 4h EMA50 trend, volume spike, and UTC 08-20 session filter.
Long when: price breaks above R1 + 4h EMA50 uptrend + volume > 1.8 * avg volume + session 08-20 UTC.
Short when: price breaks below S1 + 4h EMA50 downtrend + volume > 1.8 * avg volume + session 08-20 UTC.
Exit when: price reverts to Camarilla midpoint (PP) or touches opposite level (S1 for long, R1 for short).
Uses discrete 0.20 position size to limit drawdown. Targets 15-37 trades/year (~60-150 over 4 years) to avoid fee drag.
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
    
    # Calculate Camarilla levels from previous day (using 1d HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R1, S1, PP (pivot point)
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    
    # Align to 1h timeframe (wait for completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume spike: current volume > 1.8 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_avg)
    
    # Session filter: UTC 08-20 (inclusive)
    # prices.index is DatetimeIndex, .hour works directly
    session_hours = prices.index.hour
    in_session = (session_hours >= 8) & (session_hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for volume avg, 50 for 4h EMA
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(in_session[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.20  # Fixed position size
        
        if position == 0:
            # Flat - look for breakout with trend, volume, and session confirmation
            # Long: break above R1 + 4h EMA50 uptrend + volume spike + in session
            long_entry = (close_val > camarilla_r1_aligned[i]) and \
                       (ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]) and \
                       volume_spike[i] and \
                       in_session[i]
            # Short: break below S1 + 4h EMA50 downtrend + volume spike + in session
            short_entry = (close_val < camarilla_s1_aligned[i]) and \
                        (ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]) and \
                        volume_spike[i] and \
                        in_session[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price reverts to PP or touches S1 (contrarian exit)
            if (close_val < camarilla_pp_aligned[i]) or (close_val < camarilla_s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price reverts to PP or touches R1 (contrarian exit)
            if (close_val > camarilla_pp_aligned[i]) or (close_val > camarilla_r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0