#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation.
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > close[1] AND volume > 1.2x 20-period average volume
# Short when jaws > teeth > lips (bearish alignment) AND price < close[1] AND volume > 1.2x 20-period average volume
# Exit when Alligator lines cross (jaws > teeth for long exit, jaws < teeth for short exit)
# Uses Alligator for trend identification, volume for confirmation, and price momentum for entry timing.
# Target: 15-30 trades/year per symbol.
name = "6h_Alligator_Volume_Momentum"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Alligator (13,8,5 SMAs of median price)
    df_1d = get_htf_data(prices, '1d')
    median_price = (df_1d['high'] + df_1d['low']) / 2
    
    # Alligator lines: Jaw (13-period, 8-bar shift), Teeth (8-period, 5-bar shift), Lips (5-period, 3-bar shift)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 1d average volume for confirmation
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 20)  # Ensure Alligator and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        prev_close = close[i-1] if i > 0 else close[i]
        vol = volume[i]
        vol_ma = vol_ma_1d_aligned[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Volume confirmation: above average volume
        vol_confirm = vol > 1.2 * vol_ma
        
        if position == 0:
            # Long entry: bullish alignment + price up + volume confirmation
            if jaw_val < teeth_val < lips_val and price > prev_close and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish alignment + price down + volume confirmation
            elif jaw_val > teeth_val > lips_val and price < prev_close and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator lines cross (jaws > teeth)
            if jaw_val > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator lines cross (jaws < teeth)
            if jaw_val < teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals