#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_12hEMA34_Trend_VolumeSpike
Hypothesis: Trade 12h timeframe using Camarilla H3/L3 levels from 1d for entry, 12h EMA34 for trend filter, and 12h volume spike (>2.0x 20-bar MA) for confirmation. Enter long when price breaks above H3 AND above EMA34 AND volume spike. Enter short when price breaks below L3 AND below EMA34 AND volume spike. Exit on opposite Camarilla touch (L4/H4) or trend reversal. Uses discrete sizing 0.25 to balance return and drawdown. Target 20-50 trades/year on 12h timeframe. Works in bull/bear via Camarilla structure and trend filter.
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
    
    # Get 1d data for Camarilla pivot levels (H3, L3, H4, L4)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d
    # H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    # H4 = close + 1.5*(high-low)/2, L4 = close - 1.5*(high-low)/2
    camarilla_range = high_1d - low_1d
    h3 = close_1d + (1.1 * camarilla_range / 2)
    l3 = close_1d - (1.1 * camarilla_range / 2)
    h4 = close_1d + (1.5 * camarilla_range / 2)
    l4 = close_1d - (1.5 * camarilla_range / 2)
    
    # Align Camarilla levels to 12h timeframe (completed daily bar only)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Get 12h data for EMA34 trend filter and volume confirmation
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate EMA34 for trend filter
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 20-bar volume MA on 12h for volume spike detection
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = volume_12h > (2.0 * vol_ma_12h)
    volume_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34), volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(volume_spike_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above H3 AND above EMA34 AND volume spike
            long_setup = (close[i] > h3_aligned[i]) and \
                         (close[i] > ema_34_12h_aligned[i]) and \
                         volume_spike_12h_aligned[i]
            # Short: price breaks below L3 AND below EMA34 AND volume spike
            short_setup = (close[i] < l3_aligned[i]) and \
                          (close[i] < ema_34_12h_aligned[i]) and \
                          volume_spike_12h_aligned[i]
            
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
            # Exit: price touches L4 OR closes below EMA34
            if (close[i] <= l4_aligned[i]) or \
               (close[i] < ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches H4 OR closes above EMA34
            if (close[i] >= h4_aligned[i]) or \
               (close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_12hEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0