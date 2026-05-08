#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation
# Uses Alligator (Jaw/Teeth/Lips) to identify trend direction and strength
# Long when Lips > Teeth > Jaw (bullish alignment) + 1d EMA50 up + volume > 1.5x average
# Short when Lips < Teeth < Jaw (bearish alignment) + 1d EMA50 down + volume > 1.5x average
# Avoids whipsaws by requiring strong alignment and volume confirmation
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "6h_WilliamsAlligator_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator (13,8,5) - Smoothed Medians
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Average volume for confirmation
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for Alligator calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema50_1d_val = ema50_1d_aligned[i]
        vol_val = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: Bullish alignment (Lips > Teeth > Jaw) + 1d uptrend + volume confirmation
            if (lips_val > teeth_val > jaw_val and 
                ema50_1d_val > 0 and 
                vol_val > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Enter short: Bearish alignment (Lips < Teeth < Jaw) + 1d downtrend + volume confirmation
            elif (lips_val < teeth_val < jaw_val and 
                  ema50_1d_val < 0 and 
                  vol_val > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alignment breaks or trend reverses
            if not (lips_val > teeth_val > jaw_val) or ema50_1d_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alignment breaks or trend reverses
            if not (lips_val < teeth_val < jaw_val) or ema50_1d_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals