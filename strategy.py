#!/usr/bin/env python3
# 12h_Alligator_Jaw_Teeth_Lips_1dTrend_Volume
# Hypothesis: 12h chart strategy using Williams Alligator indicator with 1d EMA trend filter and volume confirmation.
# Alligator uses 3 SMAs (Jaw=13, Teeth=8, Lips=5) to identify trend and avoid sideways markets.
# Long when Lips > Teeth > Jaw (bullish alignment) with volume spike and price above 1d EMA50.
# Short when Lips < Teeth < Jaw (bearish alignment) with volume spike and price below 1d EMA50.
# Designed for low trade frequency (12-37/year) to avoid fee drag in bear markets.

name = "12h_Alligator_Jaw_Teeth_Lips_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d closes for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 12h data for Williams Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h data
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # Using SMA as approximation for SMMA (Smoothed Moving Average)
    jaw_12h = pd.Series(df_12h['close'].values).rolling(window=13, min_periods=13).mean().values
    teeth_12h = pd.Series(df_12h['close'].values).rolling(window=8, min_periods=8).mean().values
    lips_12h = pd.Series(df_12h['close'].values).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator components to lower timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    
    # Volume spike detection: 2x average volume (2-period = 1 day on 12h chart)
    vol_ma = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13, 2)  # Ensure we have EMA, Jaw, and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
            
            # Long: bullish alignment + volume spike + price above 1d EMA50
            if bullish_alignment and volume[i] > 2.0 * vol_ma[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment + volume spike + price below 1d EMA50
            elif bearish_alignment and volume[i] > 2.0 * vol_ma[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Alligator alignment turns bearish or price crosses below EMA50
            bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
            if bearish_alignment or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Alligator alignment turns bullish or price crosses above EMA50
            bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
            if bullish_alignment or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals