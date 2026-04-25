#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Trade 6h Camarilla R3/S3 breakouts with 1-week EMA50 trend filter and 1-day volume spike (>2.0x 20-bar MA). Uses weekly trend for stronger macro bias, reducing false signals in choppy markets. Volume spike confirms institutional interest. Discrete sizing 0.25 to balance return and drawdown. Target 12-30 trades/year on 6h timeframe. Works in bull/bear via trend filter + volume confirmation.
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
    
    # Get 1w data for HTF trend (EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w for HTF trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-bar volume MA on 1d for volume spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate Camarilla levels from previous 6h bar (for 6h entry timing)
    camarilla_range = (high - low) * 1.1 / 12.0
    camarilla_R3 = close + camarilla_range * 3.0
    camarilla_S3 = close - camarilla_range * 3.0
    
    # Shift by 1 to use only completed 6h bar for Camarilla calculation (no look-ahead)
    camarilla_R3 = np.roll(camarilla_R3, 1)
    camarilla_S3 = np.roll(camarilla_S3, 1)
    camarilla_R3[0] = np.nan
    camarilla_S3[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50), volume MA (20), and Camarilla (1)
    start_idx = max(50, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(camarilla_R3[i]) or 
            np.isnan(camarilla_S3[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + above 1w EMA50 + 1d volume spike
            long_setup = (close[i] > camarilla_R3[i]) and \
                         (close[i] > ema_50_1w_aligned[i]) and \
                         volume_spike_1d_aligned[i]
            # Short: price breaks below Camarilla S3 + below 1w EMA50 + 1d volume spike
            short_setup = (close[i] < camarilla_S3[i]) and \
                          (close[i] < ema_50_1w_aligned[i]) and \
                          volume_spike_1d_aligned[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price closes below Camarilla S3 OR below 1w EMA50
            if (close[i] < camarilla_S3[i]) or \
               (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above Camarilla R3 OR above 1w EMA50
            if (close[i] > camarilla_R3[i]) or \
               (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0