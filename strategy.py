#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H4/L4 breakout with 1d Williams %R regime filter and volume confirmation.
- Long: Close > Camarilla H4 AND Williams %R(1d) < -80 (oversold) AND volume > 1.8x 20-period avg
- Short: Close < Camarilla L4 AND Williams %R(1d) > -20 (overbought) AND volume > 1.8x 20-period avg
- Exit: Opposite Camarilla breakout OR Williams %R crosses centerline (-50)
- Uses 1d HTF for Williams %R and Camarilla levels (calculated from prior 1d bar)
- Designed for low trade frequency (12-37/year) to minimize fee drag
- Williams %R regime filter avoids buying strength/weakness, picks pullbacks in trend
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
    
    # Volume confirmation: > 1.8x 20-period average (20*6h = 5 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Williams %R for regime filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate Camarilla levels from prior 1d bar (HTF = 1d)
    # Camarilla: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 6h timeframe (use prior completed 1d bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Need 20 for volume MA, 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(williams_r_aligned[i]) or
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
        
        # Williams %R regime: < -80 oversold, > -20 overbought
        williams_oversold = williams_r_aligned[i] < -80
        williams_overbought = williams_r_aligned[i] > -20
        williams_bullish = williams_r_aligned[i] > -50  # above centerline
        williams_bearish = williams_r_aligned[i] < -50  # below centerline
        
        if position == 0:
            # Long: Camarilla H4 breakout up AND Williams %R oversold AND volume confirmation
            if breakout_up and williams_oversold and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla L4 breakout down AND Williams %R overbought AND volume confirmation
            elif breakout_down and williams_overbought and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Camarilla L4 breakout down OR Williams %R crosses above -50 (bullish regime)
            if breakout_down or williams_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Camarilla H4 breakout up OR Williams %R crosses below -50 (bearish regime)
            if breakout_up or williams_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H4L4_Breakout_1dWilliamsR_VolumeConfirm"
timeframe = "6h"
leverage = 1.0