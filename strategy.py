#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H4/L4 breakout with 1d Williams %R extreme filter and volume confirmation.
- Uses Camarilla pivot levels (H4, L4) from 1d for stronger breakout signals (wider bands than H3/L3)
- 1d Williams %R(14) as regime filter: long only when %R > -20 (not overbought), short only when %R < -80 (not oversold)
- Volume > 2.0x 20-period average for confirmation to reduce false breakouts
- Position size: 0.30 discrete level for optimal risk/reward
- Target: 25-40 trades/year on 4h timeframe (100-160 total over 4 years)
- Williams %R filter avoids counter-trend entries during extreme conditions, improving performance in both bull/bear markets
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
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for Camarilla pivot calculation and Williams %R
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (H4, L4) from prior 1d bar - stronger breakout levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    camarilla_h4 = close_1d + (rng * 1.1 / 2.0)  # H4 level
    camarilla_l4 = close_1d - (rng * 1.1 / 2.0)  # L4 level
    
    # Align Camarilla levels to 1d timeframe (using completed 1d bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 1d Williams %R(14) - momentum oscillator
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Volume MA, Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(williams_r_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_h4_aligned[i]  # Close above H4
        breakout_down = close[i] < camarilla_l4_aligned[i]  # Close below L4
        
        # Williams %R regime filter
        # Long: avoid overbought conditions (%R > -20)
        # Short: avoid oversold conditions (%R < -80)
        wr_long_filter = williams_r_aligned[i] > -20
        wr_short_filter = williams_r_aligned[i] < -80
        
        if position == 0:
            # Long: 1d Camarilla H4 breakout up AND Williams %R not overbought AND volume confirmation
            if breakout_up and wr_long_filter and volume_confirm:
                signals[i] = 0.30
                position = 1
            # Short: 1d Camarilla L4 breakout down AND Williams %R not oversold AND volume confirmation
            elif breakout_down and wr_short_filter and volume_confirm:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: 1d Camarilla L4 breakdown OR Williams %R becomes oversold
            if breakout_down or williams_r_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: 1d Camarilla H4 breakout OR Williams %R becomes overbought
            if breakout_up or williams_r_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_H4L4_Breakout_1dWilliamsR_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0