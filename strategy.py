#!/usr/bin/env python3
"""
1d_WeeklyCamarilla_H3L3_Breakout_TrendFilter_VolumeSpike
Hypothesis: On daily timeframe, trade breakouts of weekly Camarilla H3/L3 levels with 
weekly EMA34 trend filter and daily volume spike (>2.0x 20-bar MA) confirmation. 
Enter long when price > weekly H3 AND above weekly EMA34 AND volume spike. 
Enter short when price < weekly L3 AND below weekly EMA34 AND volume spike. 
Exit on opposite weekly level touch (L3 for long, H3 for short) or trend reversal. 
Uses discrete sizing 0.25. Target 15-25 trades/year on 1d timeframe. 
Weekly Camarilla structure provides robust support/resistance that works in both 
bull and bear markets, while volume spike filters breakout validity and EMA34 
ensures alignment with weekly trend.
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
    
    # Get 1w data for weekly Camarilla levels and EMA34
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (based on previous week's range)
    # H3 = Close + 1.1 * (High - Low) / 2
    # L3 = Close - 1.1 * (High - Low) / 2
    weekly_range = high_1w - low_1w
    h3_1w = close_1w + (1.1 * weekly_range / 2.0)
    l3_1w = close_1w - (1.1 * weekly_range / 2.0)
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly indicators to daily timeframe (completed weekly bar only)
    h3_1w_aligned = align_htf_to_ltf(prices, df_1w, h3_1w)
    l3_1w_aligned = align_htf_to_ltf(prices, df_1w, l3_1w)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for daily volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-bar volume MA on 1d for volume spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(h3_1w_aligned[i]) or np.isnan(l3_1w_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly H3 AND above weekly EMA34 AND volume spike
            long_setup = (close[i] > h3_1w_aligned[i]) and \
                         (close[i] > ema_34_1w_aligned[i]) and \
                         volume_spike_1d_aligned[i]
            # Short: price below weekly L3 AND below weekly EMA34 AND volume spike
            short_setup = (close[i] < l3_1w_aligned[i]) and \
                          (close[i] < ema_34_1w_aligned[i]) and \
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
            # Exit: price touches weekly L3 OR closes below weekly EMA34
            if (close[i] <= l3_1w_aligned[i]) or \
               (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches weekly H3 OR closes above weekly EMA34
            if (close[i] >= h3_1w_aligned[i]) or \
               (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyCamarilla_H3L3_Breakout_TrendFilter_VolumeSpike"
timeframe = "1d"
leverage = 1.0