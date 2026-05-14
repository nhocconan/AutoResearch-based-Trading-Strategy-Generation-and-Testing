#!/usr/bin/env python3
# 6h_Williams_Alligator_Trend_With_1d_Filter
# Hypothesis: Uses Williams Alligator (3 SMAs: Jaw/Teeth/Lips) on 6h timeframe to detect trend.
# Enters long when Lips > Teeth > Jaw (bullish alignment) and price is above Teeth with 1d trend confirmation.
# Enters short when Lips < Teeth < Jaw (bearish alignment) and price is below Teeth with 1d trend confirmation.
# Uses 1d EMA(34) as higher timeframe trend filter to avoid counter-trend trades.
# Designed for low trade frequency (target: 20-50 trades/year) with strong trend persistence.

name = "6h_Williams_Alligator_Trend_With_1d_Filter"
timeframe = "6h"
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
    
    # 6h Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3)
    # Jaw: 13-period SMMA, smoothed 8 periods ahead
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw_raw.rolling(window=8, min_periods=8).mean()
    
    # Teeth: 8-period SMMA, smoothed 5 periods ahead
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth_raw.rolling(window=5, min_periods=5).mean()
    
    # Lips: 5-period SMMA, smoothed 3 periods ahead
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips_raw.rolling(window=3, min_periods=3).mean()
    
    # 1d EMA(34) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator alignments
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long: Bullish alignment + price above Teeth + 1d uptrend (price > EMA34)
            if bullish_alignment and close[i] > teeth[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + price below Teeth + 1d downtrend (price < EMA34)
            elif bearish_alignment and close[i] < teeth[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bearish alignment or price below Jaw (trend weakness)
            if bearish_alignment or close[i] < jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bullish alignment or price above Jaw (trend weakness)
            if bullish_alignment or close[i] > jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals