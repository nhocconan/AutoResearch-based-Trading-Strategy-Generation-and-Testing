#!/usr/bin/env python3
"""
Hypothesis: 1d Williams Alligator with 1w EMA50 Trend Filter and Volume Spike
- Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5) to identify trend direction
- Long when Lips > Teeth > Jaw (bullish alignment) AND price > Alligator Jaw AND volume > 1.5x 20-period MA
- Short when Lips < Teeth < Jaw (bearish alignment) AND price < Alligator Jaw AND volume > 1.5x 20-period MA
- Exit when Alligator lines cross (trend change) OR volume drops below average
- Designed for 1d timeframe to capture medium-term trends with minimal trades
- Target: 7-25 trades/year per symbol (30-100 total over 4 years) to avoid fee drag
- Williams Alligator works in both bull and bear markets by clearly defining trend structure
- Volume confirmation prevents false breakouts in low-participation moves
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
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator on 1d data (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    
    # Alligator Jaw (13-period SMMA, 8 bars ahead)
    jaw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # SMMA shift
    
    # Alligator Teeth (8-period SMMA, 5 bars ahead)
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # SMMA shift
    
    # Alligator Lips (5-period SMMA, 3 bars ahead)
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # SMMA shift
    
    # Align Alligator lines to 1d timeframe (already aligned via get_htf_data)
    # But we need to ensure proper alignment for HTF multi-timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)  # need EMA50_1w, Alligator, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
            
            # Long: Bullish alignment AND price > Jaw AND volume spike
            if bullish and close[i] > jaw_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment AND price < Jaw AND volume spike
            elif bearish and close[i] < jaw_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator lines cross (trend change) OR volume drops below average
            exit_signal = False
            if position == 1:
                # Exit long when bearish alignment OR price < Jaw
                bearish = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
                if bearish or close[i] < jaw_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when bullish alignment OR price > Jaw
                bullish = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
                if bullish or close[i] > jaw_aligned[i]:
                    exit_signal = True
            
            # Additional exit: low volume (loss of momentum)
            if volume[i] < vol_ma[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Williams_Alligator_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0