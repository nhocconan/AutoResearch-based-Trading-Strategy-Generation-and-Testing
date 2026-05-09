#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Williams_Alligator_Trend_1wTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator lines (13, 8, 5 SMAs with offsets)
    close_1w = df_1w['close'].values
    jaw = pd.Series(close_1w).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close_1w).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close_1w).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align weekly Alligator lines to daily timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Daily price-based entry conditions
    # Williams Alligator: lips > teeth > jaw = uptrend, lips < teeth < jaw = downtrend
    # Enter on alignment with price confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for Alligator calculation
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw (bullish alignment) and price > Lips
            if lips_val > teeth_val > jaw_val and close[i] > lips_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaw (bearish alignment) and price < Lips
            elif lips_val < teeth_val < jaw_val and close[i] < lips_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Lips < Teeth (loss of bullish alignment) or price < Jaw
            if lips_val < teeth_val or close[i] < jaw_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Lips > Teeth (loss of bearish alignment) or price > Jaw
            if lips_val > teeth_val or close[i] > jaw_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals