#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation
# Long when price > Alligator's Jaw (teeth) and Green > Red lines (bullish alignment) with volume > 1.5x 20-period average
# Short when price < Alligator's Jaw (teeth) and Red > Green lines (bearish alignment) with volume > 1.5x 20-period average
# Exit when price crosses Alligator's Jaw (teeth) or alignment breaks
# Uses 1d EMA50 as trend filter to avoid counter-trend trades
# Williams Alligator: Jaw (SMMA13,8), Teeth (SMMA8,5), Lips (SMMA5,3)
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee decay

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h and 1d data ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Alligator components (SMMA = Smoothed Moving Average)
    def smma(arr, period):
        """Smoothed Moving Average - similar to EMA but with different smoothing"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    close_6h = df_6h['close'].values
    jaw = smma(close_6h, 13)   # Blue line (SMMA13,8)
    teeth = smma(close_6h, 8)  # Red line (SMMA8,5)
    lips = smma(close_6h, 5)   # Green line (SMMA5,3)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate 6h volume average (20-period)
    vol_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (need at least 13 for Jaw)
    start = 13
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_6h_current = volume[i]  # Current 6h volume
        
        if position == 0:
            # Bullish alignment: Green > Red > Blue (Lips > Teeth > Jaw)
            bullish_alignment = (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i])
            # Bearish alignment: Red > Green > Blue (Teeth > Lips > Jaw)
            bearish_alignment = (teeth_aligned[i] > lips_aligned[i] > jaw_aligned[i])
            
            # Long setup: price above Jaw, bullish alignment, volume confirmation, price above 1d EMA50
            if (price > jaw_aligned[i] and 
                bullish_alignment and
                vol_6h_current > 1.5 * vol_ma_6h_aligned[i] and
                price > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: price below Jaw, bearish alignment, volume confirmation, price below 1d EMA50
            elif (price < jaw_aligned[i] and 
                  bearish_alignment and
                  vol_6h_current > 1.5 * vol_ma_6h_aligned[i] and
                  price < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Jaw OR alignment breaks
            if (price < jaw_aligned[i] or 
                not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Jaw OR alignment breaks
            if (price > jaw_aligned[i] or 
                not (teeth_aligned[i] > lips_aligned[i] > jaw_aligned[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0