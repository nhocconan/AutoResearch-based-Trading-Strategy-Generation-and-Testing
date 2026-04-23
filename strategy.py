#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA50 trend filter and volume confirmation.
- Long: Close > Camarilla H3 AND price > 12h EMA50 AND volume > 1.5x 20-period avg
- Short: Close < Camarilla L3 AND price < 12h EMA50 AND volume > 1.5x 20-period avg
- Exit: Opposite Camarilla breakout OR price crosses 12h EMA50
- Uses 12h HTF for EMA50 and Camarilla levels (calculated from prior 12h bar)
- Designed for low trade frequency (20-50/year) to minimize fee drag
- Works in bull (buy breakouts above H3) and bear (sell breakdowns below L3)
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
    
    # Calculate 12h EMA50 for trend filter (HTF = 12h)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from prior 12h bar (HTF = 12h)
    # Camarilla: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_arr = df_12h['close'].values
    
    camarilla_h3 = close_12h_arr + 1.1 * (high_12h - low_12h) / 4
    camarilla_l3 = close_12h_arr - 1.1 * (high_12h - low_12h) / 4
    
    # Align Camarilla levels to 4h timeframe (use prior completed 12h bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Camarilla breakout signals (using current close vs prior levels)
        breakout_up = close[i] > camarilla_h3_aligned[i-1]  # Close above prior H3
        breakout_down = close[i] < camarilla_l3_aligned[i-1]  # Close below prior L3
        
        if position == 0:
            # Long: Camarilla H3 breakout up AND price > 12h EMA50 AND volume confirmation
            if breakout_up and volume_confirm and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla L3 breakout down AND price < 12h EMA50 AND volume confirmation
            elif breakout_down and volume_confirm and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Camarilla L3 breakout down OR price < 12h EMA50 (trend flip)
            if breakout_down or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Camarilla H3 breakout up OR price > 12h EMA50 (trend flip)
            if breakout_up or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_12hEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0