#!/usr/bin/env python3
"""
12h Alligator + Volume Spike + Trend Filter
Long when Alligator jaws (blue line) crosses above teeth (red line) with volume > 1.5x 20-period average and price > 12h EMA34.
Short when jaws crosses below teeth with volume > 1.5x average and price < EMA34.
Exit on opposite crossover or when price crosses the lips (green line).
Designed for 12h to capture trends with low trade frequency (~15-30/year) and avoid whipsaws in ranging markets.
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
    
    # Get 12h data for Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Alligator lines: SMAs of median price (hl2) with specific periods and shifts
    hl2 = (df_12h['high'] + df_12h['low']) / 2
    jaws = hl2.rolling(window=13, min_periods=13).mean().shift(8)   # Blue line
    teeth = hl2.rolling(window=8, min_periods=8).mean().shift(5)    # Red line
    lips = hl2.rolling(window=5, min_periods=5).mean().shift(3)     # Green line
    
    # Align to lower timeframe (12h -> 12h is identity, but we keep for consistency)
    jaws_12h = align_htf_to_ltf(prices, df_12h, jaws.values)
    teeth_12h = align_htf_to_ltf(prices, df_12h, teeth.values)
    lips_12h = align_htf_to_ltf(prices, df_12h, lips.values)
    
    # 12h EMA34 for trend filter
    ema_34 = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # Volume confirmation: 20-period volume MA on 12h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for Alligator and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(jaws_12h[i]) or np.isnan(teeth_12h[i]) or np.isnan(lips_12h[i]) or 
            np.isnan(ema_34_12h[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: jaws crosses above teeth with volume spike and price > EMA34
            if jaws_12h[i] > teeth_12h[i] and jaws_12h[i-1] <= teeth_12h[i-1] and \
               vol > 1.5 * vol_ma and price > ema_34_12h[i]:
                signals[i] = 0.25
                position = 1
            # Short: jaws crosses below teeth with volume spike and price < EMA34
            elif jaws_12h[i] < teeth_12h[i] and jaws_12h[i-1] >= teeth_12h[i-1] and \
                 vol > 1.5 * vol_ma and price < ema_34_12h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: jaws crosses below teeth OR price crosses below lips
            if jaws_12h[i] < teeth_12h[i] and jaws_12h[i-1] >= teeth_12h[i-1] or \
               price < lips_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: jaws crosses above teeth OR price crosses above lips
            if jaws_12h[i] > teeth_12h[i] and jaws_12h[i-1] <= teeth_12h[i-1] or \
               price > lips_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Alligator_Volume_EMA34"
timeframe = "12h"
leverage = 1.0