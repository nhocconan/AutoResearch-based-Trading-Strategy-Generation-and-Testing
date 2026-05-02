#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 12h EMA50 trend filter and volume confirmation
# Williams Alligator (Jaw, Teeth, Lips) identifies trend absence (all lines intertwined) vs presence (diverged)
# Trend exists when Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
# 12h EMA50 provides higher timeframe trend alignment to reduce whipsaw
# Volume spike confirms institutional participation during trend initiation
# Works in both bull and bear markets by following 12h EMA50 trend direction
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Discrete position sizing: 0.25 (25% of capital) balances opportunity and cost

name = "6h_Williams_Alligator_12hEMA50_Volume"
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
    
    # Calculate 12h Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 35:  # Need at least 35 bars for SMMA(13,8)
        return np.zeros(n)
    
    median_12h = (df_12h['high'].values + df_12h['low'].values) / 2.0
    
    # Williams Alligator: three smoothed moving averages
    # Jaw: SMMA(median, 13, 8) - Blue line
    # Teeth: SMMA(median, 8, 5) - Red line  
    # Lips: SMMA(median, 5, 3) - Green line
    def smma(source, period, shift):
        # Smoothed Moving Average: EMA-like but with different smoothing
        # First value: SMA(period)
        # Subsequent: (prev*(period-1) + current) / period
        result = np.full_like(source, np.nan)
        if len(source) < period:
            return result
        # Initial SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent SMMA values
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        # Shift the result
        result = np.roll(result, shift)
        result[:shift] = np.nan
        return result
    
    jaw = smma(median_12h, 13, 8)
    teeth = smma(median_12h, 8, 5)
    lips = smma(median_12h, 5, 3)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish trend: Lips > Teeth > Jaw (Alligator awake, eating up)
            # Bearish trend: Lips < Teeth < Jaw (Alligator awake, eating down)
            # Long entry: bullish alignment + price > 12h EMA50 + volume spike
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish alignment + price < 12h EMA50 + volume spike
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: trend turns bearish OR price falls below 12h EMA50
            if (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] or 
                close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: trend turns bullish OR price rises above 12h EMA50
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] or 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals