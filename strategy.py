#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Camarilla R1/S1 breakout on daily timeframe with 1-week EMA50 trend filter and volume spike (>2x average volume). Uses discrete position sizing (0.25) to minimize fee churn. Exits when price retests the broken Camarilla level (R1 for longs, S1 for shorts) or reverses across the 1-week EMA50. Works in both bull and bear markets by following the 1-week trend direction, confirmed by volume to avoid false breakouts. Daily timeframe reduces trade frequency to avoid fee drag while capturing multi-day trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for EMA, volume
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1-week EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load 1d data for Camarilla levels (same as primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1 levels
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 1d timeframe (1 bar delay for completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Start after warmup (need 50 for EMA, 20 for volume)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_1w_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(r1_val) or 
            np.isnan(s1_val)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 2x average volume (strong breakout)
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Long logic: price breaks above Camarilla R1 with 1w uptrend and volume confirmation
        long_condition = (close_val > r1_val) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below Camarilla S1 with 1w downtrend and volume confirmation
        short_condition = (close_val < s1_val) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: 
        # Long exit: price retests or breaks below Camarilla R1 (failed breakout) OR closes below 1w EMA (trend change)
        long_exit = (position == 1 and (close_val <= r1_val or close_val < ema_val))
        # Short exit: price retests or breaks above Camarilla S1 (failed breakout) OR closes above 1w EMA (trend change)
        short_exit = (position == -1 and (close_val >= s1_val or close_val > ema_val))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0