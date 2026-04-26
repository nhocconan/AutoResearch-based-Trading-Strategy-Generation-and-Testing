#!/usr/bin/env python3
"""
6h_ADX_Alligator_TrendFilter_v1
Hypothesis: On 6h timeframe, combining ADX (>25) for trend strength with Williams Alligator (Jaw/Teeth/Lips) alignment provides robust trend signals. Long when price > Alligator (Lips > Teeth > Jaw) and ADX > 25, short when price < Alligator (Lips < Teeth < Jaw) and ADX > 25. Uses discrete sizing (0.0, ±0.30) to balance profit potential and drawdown control. Targets 50-150 trades over 4 years (12-37/year) for optimal 6h frequency. Works in both bull and bear markets by following established trends with strength confirmation.
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
    
    # Calculate Williams Alligator on 6h
    # Jaw (Blue): 13-period SMMA smoothed 8 periods ahead
    jaw_raw = pd.Series(high).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values  # SMMA approximation
    
    # Teeth (Red): 8-period SMMA smoothed 5 periods ahead
    teeth_raw = pd.Series(high + low).rolling(window=8, min_periods=8).mean().values / 2
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values  # SMMA approximation
    
    # Lips (Green): 5-period SMMA smoothed 3 periods ahead
    lips_raw = pd.Series(high + low).rolling(window=5, min_periods=5).mean().values / 2
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values  # SMMA approximation
    
    # Calculate ADX on 6h
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR and DM
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Warmup: max of Alligator (13+8=21) and ADX (14+14=28) periods
    start_idx = 28 + 1
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(adx[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Alligator alignment signals
        # Bullish alignment: Lips > Teeth > Jaw
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        # Bearish alignment: Lips < Teeth < Jaw
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Price vs Alligator (using Lips as reference)
        price_above_alligator = close[i] > lips[i]
        price_below_alligator = close[i] < lips[i]
        
        # ADX trend strength filter
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Long: Bullish Alligator alignment + price above + strong trend
            long_signal = bullish_alignment and price_above_alligator and strong_trend
            
            # Short: Bearish Alligator alignment + price below + strong trend
            short_signal = bearish_alignment and price_below_alligator and strong_trend
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: Bearish Alligator alignment OR ADX weakens
            if bearish_alignment or adx[i] < 20:  # exit when trend weakens
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: Bullish Alligator alignment OR ADX weakens
            if bullish_alignment or adx[i] < 20:  # exit when trend weakens
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_Alligator_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0