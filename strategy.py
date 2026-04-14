#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d Trend Filter
# Uses Williams Alligator (Jaw, Teeth, Lips) on 4h for trend identification and entry signals
# 1d EMA (50) provides higher timeframe trend filter to avoid counter-trend trades
# Alligator lines converge in ranging markets and diverge in trending markets
# Entry when Lips cross Jaw/Teeth in direction of 1d trend
# Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA (50) for trend direction
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Williams Alligator on 4h data (Jaw=13, Teeth=8, Lips=5)
    # Smoothed Moving Average (SMMA) approximation using EMA
    jaw = pd.Series(close).ewm(span=13, adjust=False).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 13  # for Jaw calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: only trade in direction of 1d EMA
        above_ema = price > ema_1d_aligned[i]
        
        if position == 0:
            # Long: Lips cross above Jaw and Teeth (bullish alignment) with uptrend filter
            if lips[i] > jaw[i] and lips[i] > teeth[i] and lips[i-1] <= jaw[i-1] and above_ema:
                position = 1
                signals[i] = position_size
            # Short: Lips cross below Jaw and Teeth (bearish alignment) with downtrend filter
            elif lips[i] < jaw[i] and lips[i] < teeth[i] and lips[i-1] >= jaw[i-1] and not above_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Lips cross below Teeth (loss of bullish momentum) or trend changes
            if lips[i] < teeth[i] or price < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Lips cross above Teeth (loss of bearish momentum) or trend changes
            if lips[i] > teeth[i] or price > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Williams_Alligator_1dEMA_Trend"
timeframe = "4h"
leverage = 1.0