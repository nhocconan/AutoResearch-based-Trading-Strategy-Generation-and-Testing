#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSp
Hypothesis: Trade 4h timeframe using daily Camarilla R1/S1 breakout for entry, 
daily EMA34 for trend filter, and daily volume spike (>2.0x 20-bar MA) for confirmation. 
Enter long when price breaks above Camarilla R1 AND above daily EMA34 AND volume spike. 
Enter short when price breaks below Camarilla S1 AND below daily EMA34 AND volume spike. 
Exit on opposite Camarilla touch (S1 for long, R1 for short) or trend reversal. 
Uses discrete sizing 0.25 to balance return and drawdown. Target 20-50 trades/year on 4h timeframe. 
Works in bull/bear via daily structure and trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels, EMA34 trend filter, and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels (based on previous day's range)
    # Camarilla R1 = close + 1.1*(high-low)/12
    # Camarilla S1 = close - 1.1*(high-low)/12
    camarilla_r1_1d = close_1d + (1.1 * (high_1d - low_1d) / 12)
    camarilla_s1_1d = close_1d - (1.1 * (high_1d - low_1d) / 12)
    
    # Align Camarilla levels to 4h timeframe (completed daily bar only)
    camarilla_r1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
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
        if (np.isnan(camarilla_r1_1d_aligned[i]) or np.isnan(camarilla_s1_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND above daily EMA34 AND volume spike
            long_setup = (close[i] > camarilla_r1_1d_aligned[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_spike_1d_aligned[i]
            # Short: price breaks below Camarilla S1 AND below daily EMA34 AND volume spike
            short_setup = (close[i] < camarilla_s1_1d_aligned[i]) and \
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
            # Exit: price touches Camarilla S1 OR closes below daily EMA34
            if (close[i] <= camarilla_s1_1d_aligned[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla R1 OR closes above daily EMA34
            if (close[i] >= camarilla_r1_1d_aligned[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSp"
timeframe = "4h"
leverage = 1.0