#!/usr/bin/env python3
"""
4h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike
Hypothesis: Trade 4h Camarilla H3/L3 breakouts with 1d EMA34 trend filter and 1d volume spike (>2.0x 20-bar MA). Uses 1d for stronger trend confirmation than 4h alone, reducing false signals. Volume spike confirms institutional interest. Discrete sizing 0.25 to balance return and fee drag. Target 20-50 trades/year on 4h timeframe. Works in bull/bear via trend filter + volume confirmation + Camarilla structure.
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
    
    # Get 1d data for HTF trend (EMA34) and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA34 on 1d for HTF trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-bar volume MA on 1d for volume spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate Camarilla levels from previous 4h bar (for 4h entry timing)
    camarilla_range = (high - low) * 1.1 / 12.0
    camarilla_H3 = close + camarilla_range * 1.125  # H3 = C + (high-low)*1.1/12*1.125
    camarilla_L3 = close - camarilla_range * 1.125  # L3 = C - (high-low)*1.1/12*1.125
    
    # Shift by 1 to use only completed 4h bar for Camarilla calculation (no look-ahead)
    camarilla_H3 = np.roll(camarilla_H3, 1)
    camarilla_L3 = np.roll(camarilla_L3, 1)
    camarilla_H3[0] = np.nan
    camarilla_L3[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34), volume MA (20), and Camarilla (1)
    start_idx = max(34, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(camarilla_H3[i]) or 
            np.isnan(camarilla_L3[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla H3 + above 1d EMA34 + 1d volume spike
            long_setup = (close[i] > camarilla_H3[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_spike_1d_aligned[i]
            # Short: price breaks below Camarilla L3 + below 1d EMA34 + 1d volume spike
            short_setup = (close[i] < camarilla_L3[i]) and \
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
            # Exit: price closes below Camarilla L3 OR below 1d EMA34
            if (close[i] < camarilla_L3[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above Camarilla H3 OR above 1d EMA34
            if (close[i] > camarilla_H3[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0