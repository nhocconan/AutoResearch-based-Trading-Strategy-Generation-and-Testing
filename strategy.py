#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1wTrend_VolumeSpike
Hypothesis: Trade 12h timeframe using Camarilla pivot levels (H3, L3) from prior week for entry, 
1w EMA34 for trend filter, and 12h volume spike (>2.0x 20-bar MA) for confirmation. 
Enter long when price breaks above Camarilla H3 AND above 1w EMA34 AND volume spike. 
Enter short when price breaks below Camarilla L3 AND below 1w EMA34 AND volume spike. 
Exit on opposite Camarilla touch (L3 for long, H3 for short) or trend reversal. 
Uses discrete sizing 0.25 to balance return and drawdown. Target 12-30 trades/year on 12h timeframe. 
Camarilla H3/L3 levels represent stronger breakout points than H1/L1, reducing false signals. 
The 1w EMA34 filter ensures we only trade with the weekly trend, improving performance in both bull and bear markets. 
Volume confirmation avoids breakouts from low participation. 
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
    
    # Get 1w data for Camarilla pivot levels (prior week)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for prior week: H3, L3
    # Camarilla: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    camarilla_h3_1w = close_1w + (1.1 * (high_1w - low_1w) / 4)
    camarilla_l3_1w = close_1w - (1.1 * (high_1w - low_1w) / 4)
    
    # Align Camarilla levels to 12h timeframe (prior week's levels available at Monday 00:00 UTC)
    camarilla_h3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_1w)
    camarilla_l3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_1w)
    
    # Get 1w data for EMA34 trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 20-bar volume MA on 12h for volume spike detection
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = volume > (2.0 * vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1w EMA34 (34) and 12h volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_1w_aligned[i]) or np.isnan(camarilla_l3_1w_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla H3 AND above 1w EMA34 AND volume spike
            long_setup = (close[i] > camarilla_h3_1w_aligned[i]) and \
                         (close[i] > ema_34_1w_aligned[i]) and \
                         volume_spike_12h[i]
            # Short: price breaks below Camarilla L3 AND below 1w EMA34 AND volume spike
            short_setup = (close[i] < camarilla_l3_1w_aligned[i]) and \
                          (close[i] < ema_34_1w_aligned[i]) and \
                          volume_spike_12h[i]
            
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
            # Exit: price touches Camarilla L3 OR closes below 1w EMA34
            if (close[i] <= camarilla_l3_1w_aligned[i]) or \
               (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla H3 OR closes above 1w EMA34
            if (close[i] >= camarilla_h3_1w_aligned[i]) or \
               (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0