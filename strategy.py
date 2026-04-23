#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume confirmation
- Long when: Alligator jaws (13) < teeth (8) < lips (5) AND price > 1d EMA50 AND volume > 1.5x 20-period average
- Short when: Alligator jaws (13) > teeth (8) > lips (5) AND price < 1d EMA50 AND volume > 1.5x 20-period average
- Exit when: Alligator reverses (jaws crosses teeth) OR price crosses 1d EMA50 in opposite direction
- Uses 1d EMA50 as trend filter to avoid counter-trend trades
- Williams Alligator identifies trending vs ranging markets - effective in both bull and bear regimes
- Volume confirmation reduces false signals
- Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag on 4h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: SMAs of median price (typical price) with different periods
    # Jaws: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # Using typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    
    # Calculate SMMA (Smoothed Moving Average) - similar to Wilder's smoothing
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws = smma(typical_price, 13)  # Blue line
    teeth = smma(typical_price, 8)   # Red line
    lips = smma(typical_price, 5)    # Green line
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d EMA50 ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13, 50)  # Need 20 for volume MA, 13 for jaws, 50 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment conditions
        # Long alignment: jaws < teeth < lips (alligator eating with mouth up)
        long_align = jaws[i] < teeth[i] and teeth[i] < lips[i]
        # Short alignment: jaws > teeth > lips (alligator eating with mouth down)
        short_align = jaws[i] > teeth[i] and teeth[i] > lips[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Alligator long alignment + price > 1d EMA50 + volume confirmation
            if long_align and close[i] > ema_50_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Alligator short alignment + price < 1d EMA50 + volume confirmation
            elif short_align and close[i] < ema_50_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit conditions for long:
            # 1. Alligator reverses (jaws crosses above teeth) 
            # 2. Price crosses below 1d EMA50
            alligator_reverse = jaws[i] > teeth[i]
            price_below_ema = close[i] < ema_50_aligned[i]
            
            if alligator_reverse or price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short:
            # 1. Alligator reverses (jaws crosses below teeth)
            # 2. Price crosses above 1d EMA50
            alligator_reverse = jaws[i] < teeth[i]
            price_above_ema = close[i] > ema_50_aligned[i]
            
            if alligator_reverse or price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0