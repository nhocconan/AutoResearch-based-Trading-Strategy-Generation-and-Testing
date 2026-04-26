#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla R1/S1 breakouts on 4h with 1d EMA34 trend filter and volume spike (>2x average volume). Uses discrete position sizing (0.25) to minimize fee churn. Camarilla levels provide institutional support/resistance; breakouts with volume and HTF trend capture momentum in both bull and bear markets. Low trade frequency (~20-40/year) avoids fee drag while maintaining edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for EMA and volume
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla levels from previous 1d bar (H, L, C)
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla R1, S1, R2, S2, R3, S3, R4, S4
    camarilla_r1 = c_1d + (h_1d - l_1d) * 1.0/12
    camarilla_s1 = c_1d - (h_1d - l_1d) * 1.0/12
    camarilla_r2 = c_1d + (h_1d - l_1d) * 2.0/12
    camarilla_s2 = c_1d - (h_1d - l_1d) * 2.0/12
    camarilla_r3 = c_1d + (h_1d - l_1d) * 3.0/12
    camarilla_s3 = c_1d - (h_1d - l_1d) * 3.0/12
    camarilla_r4 = c_1d + (h_1d - l_1d) * 4.0/12
    camarilla_s4 = c_1d - (h_1d - l_1d) * 4.0/12
    
    # Align Camarilla levels (1d data needs to complete before use)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Start after warmup (need 20 for volume, 34 for EMA)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_34_1d_aligned[i]
        
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        r2 = r2_aligned[i]
        s2 = s2_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or 
            np.isnan(r1) or np.isnan(s1) or np.isnan(r2) or np.isnan(s2) or
            np.isnan(r3) or np.isnan(s3) or np.isnan(r4) or np.isnan(s4)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 2.0x average volume (strict filter)
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Long logic: price breaks above R1 with 1d uptrend and volume confirmation
        long_condition = (close_val > r1) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below S1 with 1d downtrend and volume confirmation
        short_condition = (close_val < s1) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: trend reversal (close crosses 1d EMA34) or opposite Camarilla level touch
        exit_long = close_val < ema_val
        exit_short = close_val > ema_val
        # Additional exit: touch opposite S1/R1 (mean reversion within day)
        exit_long_opposite = position == 1 and close_val < s1
        exit_short_opposite = position == -1 and close_val > r1
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif position == 1 and (exit_long or exit_long_opposite):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (exit_short or exit_short_opposite):
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

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0