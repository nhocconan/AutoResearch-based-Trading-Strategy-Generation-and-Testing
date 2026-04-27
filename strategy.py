#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Uses Camarilla pivot levels (R3, S3) from 12h timeframe for breakout entries, filtered by 1w EMA34 trend and volume spike (>2x average). Enters long when price breaks above 12h R3 AND 1w close > 1w EMA34 (uptrend) AND volume > 2x average. Enters short when price breaks below 12h S3 AND 1w close < 1w EMA34 (downtrend) AND volume > 2x average. Exits when price reverts to 12h close (mean reversion) OR trend breaks. Designed for 12h timeframe with tight entries to avoid fee drag: target 12-37 trades/year. Camarilla R3/S3 levels provide stronger breakout signals than R1/S1, reducing false breakouts. Weekly trend filter ensures alignment with major market direction, working in both bull and bear markets via volume confirmation to avoid low-conviction moves.
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
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA34 on 1w close for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla pivot levels for 12h timeframe
    # Camarilla: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    hl_range_12h = df_12h['high'].values - df_12h['low'].values
    camarilla_r3_12h = df_12h['close'].values + hl_range_12h * 1.1 / 4
    camarilla_s3_12h = df_12h['close'].values - hl_range_12h * 1.1 / 4
    
    # Align 12h indicators to 12h timeframe (no alignment needed for primary timeframe)
    camarilla_r3_12h_aligned = camarilla_r3_12h  # already on 12h timeframe
    camarilla_s3_12h_aligned = camarilla_s3_12h  # already on 12h timeframe
    
    # Volume confirmation: current volume > 2 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1w EMA34 (34), volume avg (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_r3_12h_aligned[i]) or np.isnan(camarilla_s3_12h_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_1w_val = ema_34_1w_aligned[i]
        r3_12h_val = camarilla_r3_12h_aligned[i]
        s3_12h_val = camarilla_s3_12h_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: price breakout above R3 (long) or below S3 (short) with trend and volume
            # Long: price > R3 AND 1w uptrend AND volume confirmation
            long_condition = (close_val > r3_12h_val) and (close_val > ema_1w_val) and vol_conf
            # Short: price < S3 AND 1w downtrend AND volume confirmation
            short_condition = (close_val < s3_12h_val) and (close_val < ema_1w_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to 12h close (mean reversion) OR trend breaks
            close_12h_aligned = df_12h['close'].values  # already aligned (same timeframe)
            exit_condition = (close_val <= close_12h_aligned[i]) or (close_val < ema_1w_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to 12h close (mean reversion) OR trend breaks
            close_12h_aligned = df_12h['close'].values  # already aligned (same timeframe)
            exit_condition = (close_val >= close_12h_aligned[i]) or (close_val > ema_1w_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0