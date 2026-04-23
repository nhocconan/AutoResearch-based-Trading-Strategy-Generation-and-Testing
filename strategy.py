#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1w EMA50 trend filter and volume spike confirmation.
- Long: Close > Camarilla H3 AND price > 1w EMA50 AND volume > 2.0x 24-period avg
- Short: Close < Camarilla L3 AND price < 1w EMA50 AND volume > 2.0x 24-period avg
- Exit: Opposite Camarilla breakout OR price crosses 1w EMA50
- Uses 1w HTF for EMA50 and Camarilla levels (calculated from prior 1w bar)
- Designed for low trade frequency (12-37/year) to minimize fee drag
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
    
    # Volume confirmation: > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate 1w EMA50 for trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from prior 1w bar (HTF = 1w)
    # Camarilla: H3 = close + 1.25*(high-low), L3 = close - 1.25*(high-low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    camarilla_h3 = close_1w_arr + 1.25 * (high_1w - low_1w)
    camarilla_l3 = close_1w_arr - 1.25 * (high_1w - low_1w)
    
    # Align Camarilla levels to 12h timeframe (use prior completed 1w bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24)  # Need 50 for EMA, 24 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Camarilla breakout signals (using current close vs prior levels)
        breakout_up = close[i] > camarilla_h3_aligned[i-1]  # Close above prior H3
        breakout_down = close[i] < camarilla_l3_aligned[i-1]  # Close below prior L3
        
        if position == 0:
            # Long: Camarilla H3 breakout up AND price > 1w EMA50 AND volume confirmation
            if breakout_up and volume_confirm and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla L3 breakout down AND price < 1w EMA50 AND volume confirmation
            elif breakout_down and volume_confirm and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Camarilla L3 breakout down OR price < 1w EMA50 (trend flip)
            if breakout_down or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Camarilla H3 breakout up OR price > 1w EMA50 (trend flip)
            if breakout_up or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0