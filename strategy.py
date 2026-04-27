#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: Uses Camarilla pivot levels (R1, S1) from 1d timeframe for breakout entries, filtered by 1w EMA50 trend and volume spike (>2x average). Enters long when price breaks above 1d R1 AND 1w close > 1w EMA50 (uptrend) AND volume > 2x average. Enters short when price breaks below 1d S1 AND 1w close < 1w EMA50 (downtrend) AND volume > 2x average. Exits when price reverts to 1d close (mean reversion) OR trend breaks. Designed for 1d timeframe with tight entries to avoid fee drag: target 30-100 total trades over 4 years (7-25/year). Camarilla levels provide high-probability reversal/breakout points, and volume confirmation avoids low-conviction moves. Works in both bull and bear markets via 1w trend filter and volume confirmation to avoid false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA50 on 1w close for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla pivot levels for 1d timeframe
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    hl_range_1d = df_1d['high'].values - df_1d['low'].values
    camarilla_r1_1d = df_1d['close'].values + hl_range_1d * 1.1 / 12
    camarilla_s1_1d = df_1d['close'].values - hl_range_1d * 1.1 / 12
    
    # Align 1d indicators to 1d timeframe (no alignment needed as primary TF is 1d)
    # But we still use align_htf_to_ltf for proper completed-bar timing
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    camarilla_r1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    # Volume confirmation: current volume > 2 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1w EMA50 (50), 1d data (0), volume avg (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r1_1d_aligned[i]) or np.isnan(camarilla_s1_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_1w_val = ema_50_1w_aligned[i]
        r1_1d_val = camarilla_r1_1d_aligned[i]
        s1_1d_val = camarilla_s1_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: price breakout above R1 (long) or below S1 (short) with trend and volume
            # Long: price > R1 AND 1w uptrend AND volume confirmation
            long_condition = (close_val > r1_1d_val) and (close_val > ema_1w_val) and vol_conf
            # Short: price < S1 AND 1w downtrend AND volume confirmation
            short_condition = (close_val < s1_1d_val) and (close_val < ema_1w_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to 1d close (mean reversion) OR trend breaks
            close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
            exit_condition = (close_val <= close_1d_aligned[i]) or (close_val < ema_1w_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to 1d close (mean reversion) OR trend breaks
            close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
            exit_condition = (close_val >= close_1d_aligned[i]) or (close_val > ema_1w_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0