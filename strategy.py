#!/usr/bin/env python3
"""
1h_Camarilla_R3S3_Breakout_4hTrend_VolumeConfirm_v2
Hypothesis: For 1h timeframe, use 4h Camarilla R3/S3 breakouts with 4h EMA50 trend filter and volume confirmation (>1.5x average volume). Enter only during 08-20 UTC session to avoid low-liquidity noise. Use discrete position sizing (0.20) to minimize fee churn. Exit when price retests the broken Camarilla level or reverses across 4h EMA50. Target: 15-35 trades/year per symbol by combining tight 4h breakout conditions with session filter and volume confirmation. Works in bull/bear by following 4h trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need warmup for 4h EMA50 and volume
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for HTF trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = close_4h + (high_4h - low_4h) * 1.1 / 4
    camarilla_s3 = close_4h - (high_4h - low_4h) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (1 bar delay for completed 4h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    
    # Start after warmup (need 50 for EMA, 20 for volume)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if outside session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_4h_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(r3_val) or 
            np.isnan(s3_val)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x average volume (breakout strength)
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Long logic: price breaks above Camarilla R3 with 4h uptrend and volume confirmation
        long_condition = (close_val > r3_val) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below Camarilla S3 with 4h downtrend and volume confirmation
        short_condition = (close_val < s3_val) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: 
        # Long exit: price retests or breaks below Camarilla R3 (failed breakout) OR closes below 4h EMA (trend change)
        long_exit = (position == 1 and (close_val <= r3_val or close_val < ema_val))
        # Short exit: price retests or breaks above Camarilla S3 (failed breakout) OR closes above 4h EMA (trend change)
        short_exit = (position == -1 and (close_val >= s3_val or close_val > ema_val))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
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

name = "1h_Camarilla_R3S3_Breakout_4hTrend_VolumeConfirm_v2"
timeframe = "1h"
leverage = 1.0