#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 12h Williams Alligator and 1d trend filter.
# Williams Alligator uses smoothed medians (Jaws, Teeth, Lips) to detect trends.
# Long: Price above all three lines (bullish alignment) + price > 1d EMA50.
# Short: Price below all three lines (bearish alignment) + price < 1d EMA50.
# Exit: Price crosses back through the middle line (Teeth).
# Uses 12h for trend structure (Alligator), 1d for trend filter, 6h for entry/exit.
# Williams Alligator is less common than basic MAs, offering potential edge.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    median_12h = (df_12h['high'].values + df_12h['low'].values) / 2.0
    
    # Williams Alligator lines (all use SMMA-like smoothing via EMA with specific periods)
    # Jaws: 13-period SMMA shifted 8 bars ahead
    # Teeth: 8-period SMMA shifted 5 bars ahead  
    # Lips: 5-period SMMA shifted 3 bars ahead
    jaws = pd.Series(median_12h).ewm(span=13, adjust=False).mean().values
    teeth = pd.Series(median_12h).ewm(span=8, adjust=False).mean().values
    lips = pd.Series(median_12h).ewm(span=5, adjust=False).mean().values
    
    # Apply the forward shifts (Alligator specific)
    jaws = np.roll(jaws, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set invalid values from roll to NaN
    jaws[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h Alligator lines to 6h
    jaws_aligned = align_htf_to_ltf(prices, df_12h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Align 1d EMA50 to 6h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(13, n):  # Start after max lookback
        # Skip if any required data is not ready
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw = jaws_aligned[i]
        tooth = teeth_aligned[i]
        lip = lips_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaws (all above)
            bullish_align = (lip > tooth) and (tooth > jaw)
            # Bearish alignment: Lips < Teeth < Jaws (all below)
            bearish_align = (lip < tooth) and (tooth < jaw)
            
            # Long: bullish alignment + above EMA50
            if bullish_align and (price > ema_trend):
                position = 1
                signals[i] = position_size
            # Short: bearish alignment + below EMA50
            elif bearish_align and (price < ema_trend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Teeth (middle line) or below EMA50
            if (price < tooth) or (price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Teeth (middle line) or above EMA50
            if (price > tooth) or (price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_1d_Williams_Alligator_EMA"
timeframe = "6h"
leverage = 1.0