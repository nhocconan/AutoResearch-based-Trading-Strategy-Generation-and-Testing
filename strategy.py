#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator breakout with 1w EMA50 trend filter and volume confirmation
# Uses 12h primary timeframe for signal generation with Williams Alligator (jaw/teeth/lips)
# 1w EMA50 trend filter provides higher timeframe bias (price > EMA50 for longs, < for shorts)
# Volume confirmation (1.8x 30-period average) filters for strong participation
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Williams Alligator identifies trend presence and direction via smoothed medians
# Works in both bull and bear markets by only trading in direction of 1w trend

name = "12h_WilliamsAlligator_1wEMA50_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Williams Alligator on 12h data
    # Jaw: 13-period SMMA smoothed by 8 periods
    # Teeth: 8-period SMMA smoothed by 5 periods  
    # Lips: 5-period SMMA smoothed by 3 periods
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Jaw (13, 8)
    jaw_raw = smma(median_price, 13)
    jaw = smma(jaw_raw, 8)
    
    # Teeth (8, 5)
    teeth_raw = smma(median_price, 8)
    teeth = smma(teeth_raw, 5)
    
    # Lips (5, 3)
    lips_raw = smma(median_price, 5)
    lips = smma(lips_raw, 3)
    
    # Align Alligator components to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw)  # Already on 12h, no alignment needed but keep pattern
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth)
    lips_aligned = align_htf_to_ltf(prices, prices, lips)
    
    # Volume confirmation (1.8x 30-period average on 12h data)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator and volume calculations)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
            # Long: Bullish alignment + volume spike + price > 1w EMA50
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + volume spike + price < 1w EMA50
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bearish Alligator alignment or price < 1w EMA50
            if (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] or 
                close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bullish Alligator alignment or price > 1w EMA50
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] or 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals