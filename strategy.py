#!/usr/bin/env python3
"""
12h strategy combining 1d Williams Alligator with 12h price momentum and volume confirmation.
Williams Alligator: Jaw (13-period SMA shifted 8), Teeth (8-period SMA shifted 5), Lips (5-period SMA shifted 3)
Trend condition: Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
Entry: 12h price crosses above/below 12-period EMA with volume > 1.5x 20-period volume MA in trend direction
Exit: Price crosses back below/above the 12-period EMA
Position size: 0.25
Designed for 12h timeframe with strict trend and momentum filters to limit trades to 50-150 total over 4 years
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
    
    # Get 1-day data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams Alligator components on 1d data
    # Jaw: 13-period SMMA shifted 8 bars
    jaw_13 = pd.Series(df_1d['close']).rolling(window=13, min_periods=13).mean()
    jaw = jaw_13.shift(8)  # shift 8 bars forward
    
    # Teeth: 8-period SMMA shifted 5 bars
    teeth_8 = pd.Series(df_1d['close']).rolling(window=8, min_periods=8).mean()
    teeth = teeth_8.shift(5)  # shift 5 bars forward
    
    # Lips: 5-period SMMA shifted 3 bars
    lips_5 = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).mean()
    lips = lips_5.shift(3)  # shift 3 bars forward
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw.values)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth.values)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips.values)
    
    # 12h EMA12 for momentum
    ema_12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 34  # warmup for Alligator (max shift 8 + jaw period 13)
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_12[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_val = ema_12[i]
        
        # Bullish Alligator alignment: Lips > Teeth > Jaw
        bullish_align = lips_val > teeth_val and teeth_val > jaw_val
        # Bearish Alligator alignment: Lips < Teeth < Jaw
        bearish_align = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Look for entries with Alligator trend confirmation
            # Long: price crosses above EMA12 + volume spike + bullish Alligator
            if price > ema_val and vol > 1.5 * vol_ma and bullish_align:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below EMA12 + volume spike + bearish Alligator
            elif price < ema_val and vol > 1.5 * vol_ma and bearish_align:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses back below EMA12
            if price < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses back above EMA12
            if price > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_EMA12_Volume"
timeframe = "12h"
leverage = 1.0