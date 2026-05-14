#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation.
# The Williams Alligator uses SMAs (Jaw=13, Teeth=8, Lips=5) to identify trends.
# In uptrends: Lips > Teeth > Jaw. In downtrends: Lips < Teeth < Jaw.
# Combined with 1d EMA trend filter and volume spikes, it filters false signals.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
# Works in both bull and bear markets by using 1d trend filter to avoid counter-trend trades.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA(20) for 1d trend filter
    ema20_1d = np.zeros(len(close_1d))
    ema_multiplier = 2 / (20 + 1)
    ema20_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema20_1d[i] = (close_1d[i] - ema20_1d[i-1]) * ema_multiplier + ema20_1d[i-1]
    
    # Align 1d EMA to 4h timeframe
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Williams Alligator on 4h timeframe
    # Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    def smoothed_mma(arr, period):
        """Smoothed Moving Average (SMMA) - similar to Wilder's smoothing"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full(len(arr), np.nan)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smoothed_mma(close, 13)  # Jaw: 13-period SMMA
    teeth = smoothed_mma(close, 8)  # Teeth: 8-period SMMA
    lips = smoothed_mma(close, 5)   # Lips: 5-period SMMA
    
    # Average volume (20-period = 20*4h = 10 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema20_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema20_1d_aligned[i]
        
        # Alligator values
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (aligned up) + above 1d EMA20 + volume confirmation
            if (lips_val > teeth_val and 
                teeth_val > jaw_val and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Lips < Teeth < Jaw (aligned down) + below 1d EMA20 + volume confirmation
            elif (lips_val < teeth_val and 
                  teeth_val < jaw_val and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator lines cross or price breaks below 1d EMA
            if (lips_val <= teeth_val or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator lines cross or price breaks above 1d EMA
            if (lips_val >= teeth_val or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_WilliamsAlligator_Trend_Volume"
timeframe = "4h"
leverage = 1.0