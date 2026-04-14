#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w Trend Filter
# Uses Williams Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5) to detect trend direction
# 1w EMA (50) provides higher timeframe trend filter to avoid counter-trend trades
# Alligator works in both bull/bear markets by identifying when trends are sleeping or awakening
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA (50) for trend direction
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Williams Alligator on 1d data
    # Jaw (blue line): 13-period SMMA smoothed 8 periods ahead
    # Teeth (red line): 8-period SMMA smoothed 5 periods ahead  
    # Lips (green line): 5-period SMMA smoothed 3 periods ahead
    # Using SMA as approximation for SMMA for computational efficiency
    
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max of all periods)
    start = 20  # covers all rolling windows
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: only trade in direction of 1w EMA
        above_ema = price > ema_1w_aligned[i]
        
        # Alligator signals: 
        # Awakening (trend starting): Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
        # Sleeping (no trend): lines are intertwined
        
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        bullish_alignment = lips_above_teeth and teeth_above_jaw
        
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        bearish_alignment = lips_below_teeth and teeth_below_jaw
        
        if position == 0:
            # Long: bullish alignment with uptrend filter
            if bullish_alignment and above_ema:
                position = 1
                signals[i] = position_size
            # Short: bearish alignment with downtrend filter
            elif bearish_alignment and not above_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish alignment or trend changes
            if bearish_alignment or not above_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: bullish alignment or trend changes
            if bullish_alignment or above_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Williams_Alligator_1wEMA_Trend"
timeframe = "1d"
leverage = 1.0