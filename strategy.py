#!/usr/bin/env python3
"""
6h Williams Alligator with 1d Trend Filter
Long when Alligator aligns bullish (jaw < teeth < lips) and 1d close > 1d EMA50
Short when Alligator aligns bearish (jaw > teeth > lips) and 1d close < 1d EMA50
Exit when alignment breaks
Williams Alligator identifies trend phases; 1d filter avoids counter-trend trades
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_williams_alligator_1d_trend_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === Williams Alligator (13,8,5 smoothed with 8,5,3) ===
    # Jaw (blue): 13-period SMMA, smoothed 8 periods
    # Teeth (red): 8-period SMMA, smoothed 5 periods  
    # Lips (green): 5-period SMMA, smoothed 3 periods
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev SMMA * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate SMMA components
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Apply smoothing
    jaw = smma(jaw_raw, 8)
    teeth = smma(teeth_raw, 5)
    lips = smma(lips_raw, 3)
    
    # === 1d Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Check Alligator alignment
        bullish_align = jaw[i] < teeth[i] and teeth[i] < lips[i]
        bearish_align = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        if position == 1:  # Long position
            # Exit: alignment breaks or trend fails
            if not bullish_align or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: alignment breaks or trend fails
            if not bearish_align or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Enter on aligned Alligator with 1d trend confirmation
            if bullish_align and close[i] > ema_50_1d_aligned[i]:
                # Bullish alignment + above 1d EMA50 -> long
                position = 1
                signals[i] = 0.25
            elif bearish_align and close[i] < ema_50_1d_aligned[i]:
                # Bearish alignment + below 1d EMA50 -> short
                position = -1
                signals[i] = -0.25
    
    return signals