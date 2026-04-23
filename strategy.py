#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA34 Trend Filter and Volume Spike
- Uses Williams Alligator (Jaw/Teeth/Lips) for trend identification on 12h timeframe
- Entry: Price > Lips AND Lips > Teeth AND Teeth > Jaw (bullish alignment) OR inverse for bearish
- Filters: 1d EMA34 trend alignment + volume > 1.5x 20-period MA
- Exit: Loss of Alligator alignment OR loss of 1d EMA34 trend
- Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years)
- Williams Alligator catches sustained trends while filtering choppy markets
- Works in both bull and bear markets via trend filter and volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    median_12h = (df_12h['high'].values + df_12h['low'].values) / 2
    
    # Jaw: Blue line (13-period SMMA, shifted 8 bars)
    jaw = pd.Series(median_12h).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth: Red line (8-period SMMA, shifted 5 bars)
    teeth = pd.Series(median_12h).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips: Green line (5-period SMMA, shifted 3 bars)
    lips = pd.Series(median_12h).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Align Alligator lines to primary timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 34, 20)  # need Alligator, EMA34_1d, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish Alligator alignment (Lips > Teeth > Jaw) AND price > 1d EMA34 AND volume spike
            if (lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment (Lips < Teeth < Jaw) AND price < 1d EMA34 AND volume spike
            elif (lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Loss of Alligator alignment OR loss of 1d EMA34 trend
            exit_signal = False
            if position == 1:
                # Exit long when bearish alignment OR price < 1d EMA34
                if not (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]) or \
                   close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when bullish alignment OR price > 1d EMA34
                if not (lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]) or \
                   close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0