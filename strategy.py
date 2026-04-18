#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter.
# Long when: Alligator lines aligned bullish (jaw < teeth < lips) and price above 1d EMA(50)
# Short when: Alligator lines aligned bearish (jaw > teeth > lips) and price below 1d EMA(50)
# Exit when Alligator alignment breaks or price crosses 1d EMA(50) in opposite direction.
# Williams Alligator uses smoothed median prices (SMMA) for jaw(13,8), teeth(8,5), lips(5,3).
# Designed to catch trends with low noise, suitable for 6h timeframe with ~15-30 trades/year.
name = "6h_WilliamsAlligator_1dEMA50"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate median price (typical price simplified)
    median_price = (high + low) / 2.0
    
    # Williams Alligator: three SMMA lines
    def smma(arr, period):
        """Smoothed Moving Average - similar to Wilder's smoothing"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_smma * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)  # Blue line
    teeth = smma(median_price, 8)  # Red line
    lips = smma(median_price, 5)   # Green line
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after Alligator warmup (max period 13) + 1d EMA warmup
    start_idx = max(13, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema_50 = ema_50_1d_aligned[i]
        
        # Alligator alignment conditions
        bullish_aligned = jaw_val < teeth_val < lips_val
        bearish_aligned = jaw_val > teeth_val > lips_val
        
        if position == 0:
            # Long: bullish alignment and price above 1d EMA50
            if bullish_aligned and price > ema_50:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment and price below 1d EMA50
            elif bearish_aligned and price < ema_50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: alignment breaks bearish or price crosses below 1d EMA50
            if not bullish_aligned or price < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: alignment breaks bullish or price crosses above 1d EMA50
            if not bearish_aligned or price > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals