#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H4/L4 breakout with 1d EMA34 trend filter and volume confirmation.
- Long: Close > Camarilla H4 AND price > 1d EMA34 AND volume > 1.8x 20-period avg
- Short: Close < Camarilla L4 AND price < 1d EMA34 AND volume > 1.8x 20-period avg
- Exit: Opposite Camarilla breakout OR price crosses 1d EMA34
- Uses 1d HTF for EMA34 and Camarilla levels (calculated from prior 1d bar)
- Designed for low trade frequency (19-50/year) to minimize fee drag
- Works in bull (buy breakouts above H4) and bear (sell breakdowns below L4)
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
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA34 for trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d bar (HTF = 1d)
    # Camarilla: H4 = close + 1.5*(high-low)/2, L4 = close - 1.5*(high-low)/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    camarilla_h4 = close_1d_arr + 1.5 * (high_1d - low_1d) / 2
    camarilla_l4 = close_1d_arr - 1.5 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 4h timeframe (use prior completed 1d bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 34 for EMA, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        # Camarilla breakout signals (using current close vs prior levels)
        breakout_up = close[i] > camarilla_h4_aligned[i-1]  # Close above prior H4
        breakout_down = close[i] < camarilla_l4_aligned[i-1]  # Close below prior L4
        
        if position == 0:
            # Long: Camarilla H4 breakout up AND price > 1d EMA34 AND volume confirmation
            if breakout_up and volume_confirm and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.30
                position = 1
            # Short: Camarilla L4 breakout down AND price < 1d EMA34 AND volume confirmation
            elif breakout_down and volume_confirm and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: Camarilla L4 breakout down OR price < 1d EMA34 (trend flip)
            if breakout_down or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: Camarilla H4 breakout up OR price > 1d EMA34 (trend flip)
            if breakout_up or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_H4L4_Breakout_1dEMA34_VolumeConfirm"
timeframe = "4h"
leverage = 1.0