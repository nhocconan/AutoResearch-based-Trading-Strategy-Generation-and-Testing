#!/usr/bin/env python3
# 6h_Williams_Alligator_Triple_Cross
# Hypothesis: Williams Alligator (Jaw 13, Teeth 8, Lips 5 SMAs) on 6h timeframe with 1d trend filter.
# Long when Lips > Teeth > Jaw (bullish alignment) and price above 1d EMA50.
# Short when Lips < Teeth < Jaw (bearish alignment) and price below 1d EMA50.
# Exit when alignment breaks or price crosses 1d EMA50 in opposite direction.
# Designed to catch trends in both bull and bear markets by aligning with higher timeframe trend.
# Target: 15-35 trades/year (~60-140 total over 4 years) to stay within optimal trade frequency for 6h.

name = "6h_Williams_Alligator_Triple_Cross"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator on 6h: Jaw (13), Teeth (8), Lips (5) - all SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # 1d EMA trend filter (50-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long: bullish alignment and price above 1d EMA50
            if bullish_alignment and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment and price below 1d EMA50
            elif bearish_alignment and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: bearish alignment OR price crosses below 1d EMA50
            if bearish_alignment or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: bullish alignment OR price crosses above 1d EMA50
            if bullish_alignment or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals