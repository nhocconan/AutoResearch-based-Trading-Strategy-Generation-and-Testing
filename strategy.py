#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d Trend Filter
# Uses Williams Alligator (Jaw/Teeth/Lips) on 12h timeframe to identify trend direction
# Filters trades using 1d EMA200 to ensure alignment with higher timeframe trend
# Williams Alligator consists of three smoothed moving averages:
#   Jaw (13-period, shifted 8 bars forward) - Blue line
#   Teeth (8-period, shifted 5 bars forward) - Red line  
#   Lips (5-period, shifted 3 bars forward) - Green line
# Entry signals:
#   Long: Lips > Teeth > Jaw (bullish alignment) and price above 1d EMA200
#   Short: Lips < Teeth < Jaw (bearish alignment) and price below 1d EMA200
# Exit: When Alligator lines intertwine (Lips crosses Teeth) indicating trend weakness
# Designed for low frequency (~15-25 trades/year) to minimize fee drag
# Works in both bull and bear markets by following the dominant trend

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for EMA200 filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate 1d EMA200
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Williams Alligator components on 12h data
    # Jaw: 13-period SMMA, shifted 8 bars
    jaw = pd.Series(high).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMMA, shifted 5 bars
    teeth = pd.Series(low).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMMA, shifted 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max shift is 8)
    start = 200  # for 1d EMA200 calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Check Alligator alignment
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        bullish_alignment = lips_above_teeth and teeth_above_jaw
        
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        bearish_alignment = lips_below_teeth and teeth_below_jaw
        
        # Check 1d EMA200 filter
        price_above_ema200 = price > ema_200_1d_aligned[i]
        price_below_ema200 = price < ema_200_1d_aligned[i]
        
        if position == 0:
            # Long entry: bullish Alligator alignment + price above 1d EMA200
            if bullish_alignment and price_above_ema200:
                position = 1
                signals[i] = position_size
            # Short entry: bearish Alligator alignment + price below 1d EMA200
            elif bearish_alignment and price_below_ema200:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: When Alligator lines intertwine (Lips crosses below Teeth)
            if lips[i] < teeth[i]:  # Lips crossed below Teeth
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: When Alligator lines intertwine (Lips crosses above Teeth)
            if lips[i] > teeth[i]:  # Lips crossed above Teeth
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Williams_Alligator_1dEMA200_Filter"
timeframe = "12h"
leverage = 1.0