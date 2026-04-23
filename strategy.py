#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume confirmation.
- Camarilla levels (H3/L3) act as strong intraday support/resistance derived from prior day's range.
- Breakout above H3 or below L3 with volume confirmation captures institutional participation.
- 1d EMA34 ensures alignment with daily trend to avoid counter-trend whipsaws.
- Discrete position size 0.25 limits drawdown during crashes.
- Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years).
- Designed to work in both bull and bear regimes via trend filter and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for EMA34 trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 on 1d close
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d bar (H3, L3, H4, L4)
    # Camarilla: H4 = high + 1.1*(high-low)/2, H3 = high + 1.1*(high-low)/4
    #            L3 = low - 1.1*(high-low)/4, L4 = low - 1.1*(high-low)/2
    high_low_diff = high_1d - low_1d
    camarilla_h3 = high_1d + 1.1 * high_low_diff / 4
    camarilla_l3 = low_1d - 1.1 * high_low_diff / 4
    camarilla_h4 = high_1d + 1.1 * high_low_diff / 2
    camarilla_l4 = low_1d - 1.1 * high_low_diff / 2
    
    # Align Camarilla levels to 4h timeframe (using prior 1d bar to avoid look-ahead)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # volume MA, 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Close > H3 AND price above 1d EMA34 AND volume confirmation
            if close[i] > h3_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Close < L3 AND price below 1d EMA34 AND volume confirmation
            elif close[i] < l3_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < L3 OR price crosses below 1d EMA34
            if close[i] < l3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > H3 OR price crosses above 1d EMA34
            if close[i] > h3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0