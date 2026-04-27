#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_SessionFilter
Hypothesis: 1h timeframe strategy using Camarilla R1/S1 breakouts filtered by 4h EMA50 trend and volume spikes. Targets 60-150 trades over 4 years (15-37/year) with 0.20 position size. Uses 4h trend for signal direction and 1h only for entry timing. Session filter (08-20 UTC) reduces noise trades. Designed to work in both bull and bear markets by aligning with the 4h trend and using volume confirmation for momentum validation.
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
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 trend filter
    ema_50 = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Get 1d data for Camarilla R1/S1 levels (from previous completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d data for Camarilla R1/S1 levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.08)   # R1 level
    s1 = prev_close - (rng * 1.08)   # S1 level
    
    # Align Camarilla levels to 1h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Fixed position size to minimize churn
    
    # Warmup: need 4h EMA50 (50), 1d shift(1) for Camarilla, vol avg (20)
    start_idx = max(50 + 1, 1 + 1, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        sess = session_filter[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with 4h EMA50 alignment, volume confirmation, and session filter
            long_condition = (close_val > r1_val and 
                            close_val > ema_val and 
                            vol_conf and 
                            sess)
            short_condition = (close_val < s1_val and 
                             close_val < ema_val and 
                             vol_conf and 
                             sess)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h EMA50 (trend reversal)
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 4h EMA50 (trend reversal)
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0