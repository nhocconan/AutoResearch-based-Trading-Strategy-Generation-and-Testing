#!/usr/bin/env python3
"""
4h_1d_Camarilla_R1S1_Breakout_VolumeSpike
Hypothesis: Trade 4h timeframe using daily Camarilla pivot levels (R1/S1) for entries,
daily EMA34 for trend filter, and daily volume spike (>2.0x 20-bar MA) for confirmation.
Enter long when price > daily R1 AND above daily EMA34 AND volume spike.
Enter short when price < daily S1 AND below daily EMA34 AND volume spike.
Exit on opposite pivot touch or trend reversal. Uses discrete sizing 0.25.
Targets 20-50 trades/year on 4h timeframe. Camarilla levels provide structure in ranging markets,
EMA34 filters trend direction, volume spike confirms institutional interest.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for daily Camarilla pivot levels, EMA34 trend filter, and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily Camarilla pivot levels
    # R1 = Close + 1.1*(High-Low)/12, S1 = Close - 1.1*(High-Low)/12
    cam_r1_1d = close_1d + (1.1 * (high_1d - low_1d) / 12)
    cam_s1_1d = close_1d - (1.1 * (high_1d - low_1d) / 12)
    
    # Align daily Camarilla levels to 4h timeframe (completed daily bar only)
    cam_r1_1d_aligned = align_htf_to_ltf(prices, df_1d, cam_r1_1d)
    cam_s1_1d_aligned = align_htf_to_ltf(prices, df_1d, cam_s1_1d)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-bar volume MA on 1d for volume spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34), volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(cam_r1_1d_aligned[i]) or np.isnan(cam_s1_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above daily R1 AND above daily EMA34 AND volume spike
            long_setup = (close[i] > cam_r1_1d_aligned[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_spike_1d_aligned[i]
            # Short: price below daily S1 AND below daily EMA34 AND volume spike
            short_setup = (close[i] < cam_s1_1d_aligned[i]) and \
                          (close[i] < ema_34_1d_aligned[i]) and \
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
            # Exit: price touches daily S1 OR closes below daily EMA34
            if (close[i] <= cam_s1_1d_aligned[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches daily R1 OR closes above daily EMA34
            if (close[i] >= cam_r1_1d_aligned[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_1d_Camarilla_R1S1_Breakout_VolumeSpike"
timeframe = "4h"
leverage = 1.0