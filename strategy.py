#!/usr/bin/env python3
"""
6h_WilliamsAlligator_12hTrend_v1
Hypothesis: Combine Williams Alligator (JAW/TEETH/LIPS) with 12h EMA50 trend filter on 6h timeframe.
Enter long when LIPS > TEETH > JAW (bullish alignment) and price > 12h EMA50.
Enter short when LIPS < TEETH < JAW (bearish alignment) and price < 12h EMA50.
Exit when Alligator lines cross (LIPS crosses TEETH) or trend changes.
Williams Alligator uses smoothed moving averages (SMMA) with periods 13/8/5 and offsets 8/5/3.
This strategy aims to catch trends while avoiding chop via trend alignment filter.
Designed for 15-30 trades/year on 6h timeframe.
Works in bull/bear markets by following 12h EMA50 trend and filtering via Alligator alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for HTF trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Williams Alligator on 6h data (using close prices)
    # JAW (Blue): 13-period SMMA, offset 8 bars
    # TEETH (Red): 8-period SMMA, offset 5 bars  
    # LIPS (Green): 5-period SMMA, offset 3 bars
    
    def smma(data, period):
        """Smoothed Moving Average"""
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Calculate SMMA for Alligator
    jaw_raw = smma(close, 13)  # 13-period
    teeth_raw = smma(close, 8)  # 8-period
    lips_raw = smma(close, 5)   # 5-period
    
    # Apply offsets (shift right by offset bars)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    if len(jaw) > 8:
        jaw[8:] = jaw_raw[:-8]
    if len(teeth) > 5:
        teeth[5:] = teeth_raw[:-5]
    if len(lips) > 3:
        lips[3:] = lips_raw[:-3]
    
    # Align HTF indicators to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Alligator periods with offsets, 12h EMA
    start_idx = max(13+8, 8+5, 5+3, 50)  # = max(21, 13, 8, 50) = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        close_val = close[i]
        ema_50_12h_val = ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: Bullish alignment (LIPS > TEETH > JAW) and price > 12h EMA50
            long_signal = (lips_val > teeth_val) and (teeth_val > jaw_val) and (close_val > ema_50_12h_val)
            
            # Short: Bearish alignment (LIPS < TEETH < JAW) and price < 12h EMA50
            short_signal = (lips_val < teeth_val) and (teeth_val < jaw_val) and (close_val < ema_50_12h_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Alligator lines cross (LIPS crosses below TEETH) or trend changes (price < 12h EMA50)
            if (lips_val < teeth_val) or (close_val < ema_50_12h_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Alligator lines cross (LIPS crosses above TEETH) or trend changes (price > 12h EMA50)
            if (lips_val > teeth_val) or (close_val > ema_50_12h_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsAlligator_12hTrend_v1"
timeframe = "6h"
leverage = 1.0