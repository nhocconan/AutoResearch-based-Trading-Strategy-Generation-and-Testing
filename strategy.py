#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: Use 6h timeframe with Camarilla R4/S4 breakout from prior day, confirmed by 1w EMA50 trend and volume spike. Targets 12-35 trades/year to minimize fee drag. R4/S4 are strong breakout levels; trading in direction of weekly trend reduces whipsaw in bear markets. Volume confirmation ensures breakout legitimacy.
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
    
    # Camarilla levels: R4, S4 (strong breakout levels)
    camarilla_r4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align to 6h timeframe (wait for completed 1d bar)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for volume avg, 50 for 1w EMA
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above R4 + 1w EMA50 uptrend + volume spike
            long_entry = (close_val > camarilla_r4_aligned[i]) and \
                       (close_val > ema_50_1w_aligned[i]) and \
                       volume_spike[i]
            # Short: break below S4 + 1w EMA50 downtrend + volume spike
            short_entry = (close_val < camarilla_s4_aligned[i]) and \
                        (close_val < ema_50_1w_aligned[i]) and \
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
            # Long - exit when price reverts to midpoint (PP) or touches S4 (contrarian exit)
            # Calculate pivot point for exit
            camarilla_pp = (prev_high[i] + prev_low[i] + prev_close[i]) / 3 if not (np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i])) else np.nan
            camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(prev_close, camarilla_pp)) if not np.isnan(camarilla_pp) else np.array([np.nan]*n)
            exit_condition = (not np.isnan(camarilla_pp_aligned[i]) and close_val < camarilla_pp_aligned[i]) or (close_val < camarilla_s4_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price reverts to midpoint (PP) or touches R4 (contrarian exit)
            camarilla_pp = (prev_high[i] + prev_low[i] + prev_close[i]) / 3 if not (np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i])) else np.nan
            camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(prev_close, camarilla_pp)) if not np.isnan(camarilla_pp) else np.array([np.nan]*n)
            exit_condition = (not np.isnan(camarilla_pp_aligned[i]) and close_val > camarilla_pp_aligned[i]) or (close_val > camarilla_r4_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0