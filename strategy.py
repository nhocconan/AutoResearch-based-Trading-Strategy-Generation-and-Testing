#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 12h Supertrend trend filter and volume spike confirmation.
- Long: Close > Camarilla H3 AND Supertrend bullish AND volume > 2.0x 20-period avg
- Short: Close < Camarilla L3 AND Supertrend bearish AND volume > 2.0x 20-period avg
- Exit: Opposite Camarilla breakout OR Supertrend flip
- Uses 12h HTF for Supertrend (more responsive than 1d, fewer whipsaws in bear markets)
- Camarilla H3/L3 provide tighter structure than H4/L4, increasing win rate while maintaining low trade frequency
- Volume confirmation filters low-conviction moves
- Designed for 25-50 trades/year to minimize fee drag on 4h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h Supertrend for trend filter (HTF = 12h)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR
    tr1 = pd.Series(high_12h).rolling(2).apply(lambda x: x[1] - x[0], raw=True)
    tr2 = pd.Series(high_12h).rolling(2).apply(lambda x: abs(x[1] - x[0]), raw=True)
    tr3 = pd.Series(low_12h).rolling(2).apply(lambda x: abs(x[1] - x[0]), raw=True)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Supertrend calculation
    supertrend = np.full(len(close_12h), np.nan)
    direction = np.full(len(close_12h), np.nan)  # 1 for up, -1 for down
    
    # Initialize
    supertrend[atr_period-1] = upper_band[atr_period-1]
    direction[atr_period-1] = 1
    
    for i in range(atr_period, len(close_12h)):
        if close_12h[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_12h[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1 and direction[i-1] == -1:
            supertrend[i] = upper_band[i]
        elif direction[i] == -1 and direction[i-1] == 1:
            supertrend[i] = lower_band[i]
        elif direction[i] == 1:
            supertrend[i] = max(upper_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(lower_band[i], supertrend[i-1])
    
    # Align Supertrend to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # Calculate Camarilla levels from prior 12h bar (HTF = 12h)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Standard Camarilla: H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    range_12h = high_12h - low_12h
    camarilla_h3 = close_12h + 1.125 * range_12h
    camarilla_l3 = close_12h - 1.125 * range_12h
    
    # Align Camarilla levels to 4h timeframe (use prior completed 12h bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(atr_period, 20)  # Need ATR period for Supertrend, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(supertrend_aligned[i]) or
            np.isnan(direction_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Supertrend trend filter
        trend_up = direction_aligned[i] == 1
        trend_down = direction_aligned[i] == -1
        
        # Camarilla breakout signals (using current close vs prior levels)
        breakout_up = close[i] > camarilla_h3_aligned[i-1]  # Close above prior H3
        breakout_down = close[i] < camarilla_l3_aligned[i-1]  # Close below prior L3
        
        if position == 0:
            # Long: Camarilla H3 breakout up AND Supertrend bullish AND volume confirmation
            if breakout_up and trend_up and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla L3 breakout down AND Supertrend bearish AND volume confirmation
            elif breakout_down and trend_down and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Camarilla L3 breakout down OR Supertrend bearish (trend flip)
            if breakout_down or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Camarilla H3 breakout up OR Supertrend bullish (trend flip)
            if breakout_up or trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_12hSupertrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0