#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSp
Hypothesis: Trade 12h timeframe using Camarilla pivot levels (H3, L3) from prior day for entry, 
1d EMA34 for trend filter, and 12h volume spike (>2.0x 20-bar MA) for confirmation. 
Enter long when price breaks above Camarilla H3 AND above 1d EMA34 AND volume spike. 
Enter short when price breaks below Camarilla L3 AND below 1d EMA34 AND volume spike. 
Exit on opposite Camarilla touch (L3 for long, H3 for short) or trend reversal. 
Uses discrete sizing 0.25 to balance return and drawdown. Target 12-37 trades/year on 12h timeframe. 
Camarilla pivots work well in ranging markets; EMA34 filter ensures we only trade with the 1d trend; 
volume confirmation avoids false breakouts. Designed to work in both bull and bear via trend filter.
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
    
    # Get 1d data for Camarilla pivot levels (prior day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for prior day: H3, L3
    # Camarilla: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    camarilla_h3_1d = close_1d + (1.1 * (high_1d - low_1d) / 4)
    camarilla_l3_1d = close_1d - (1.1 * (high_1d - low_1d) / 4)
    
    # Align Camarilla levels to 12h timeframe (prior day's levels available at 00:00 UTC)
    camarilla_h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    # Get 1d data for EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 12h data for volume spike detection
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # Calculate 20-bar volume MA on 12h for volume spike detection
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = volume_12h > (2.0 * vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d EMA34 (34) and 12h volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_1d_aligned[i]) or np.isnan(camarilla_l3_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla H3 AND above 1d EMA34 AND volume spike
            long_setup = (close[i] > camarilla_h3_1d_aligned[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_spike_12h[i]
            # Short: price breaks below Camarilla L3 AND below 1d EMA34 AND volume spike
            short_setup = (close[i] < camarilla_l3_1d_aligned[i]) and \
                          (close[i] < ema_34_1d_aligned[i]) and \
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
            # Exit: price touches Camarilla L3 OR closes below 1d EMA34
            if (close[i] <= camarilla_l3_1d_aligned[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Camarilla H3 OR closes above 1d EMA34
            if (close[i] >= camarilla_h3_1d_aligned[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSp"
timeframe = "12h"
leverage = 1.0