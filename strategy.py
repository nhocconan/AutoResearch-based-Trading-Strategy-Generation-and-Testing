#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Williams Alligator (Jaw/Teeth/Lips) with 1-week trend filter and volume confirmation.
# Alligator uses SMAs: Jaw=13, Teeth=8, Lips=5. In trends, lines are ordered (Lips > Teeth > Jaw for uptrend).
# Combined with 1-week EMA trend filter and volume spikes, it filters false signals and captures trends.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Williams Alligator on 1-day timeframe
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
    
    jaw_1d = smoothed_mma(close_1d, 13)  # Jaw: 13-period SMMA
    teeth_1d = smoothed_mma(close_1d, 8)  # Teeth: 8-period SMMA
    lips_1d = smoothed_mma(close_1d, 5)   # Lips: 5-period SMMA
    
    # Average volume (20-period) for volume confirmation
    avg_volume_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        avg_volume_1d[i] = np.mean(volume_1d[i-20:i])
    
    # EMA(20) for 1-week trend filter
    ema20_1w = np.zeros(len(close_1w))
    if len(close_1w) >= 20:
        ema_multiplier = 2 / (20 + 1)
        ema20_1w[0] = close_1w[0]
        for i in range(1, len(close_1w)):
            ema20_1w[i] = (close_1w[i] - ema20_1w[i-1]) * ema_multiplier + ema20_1w[i-1]
    else:
        ema20_1w[:] = np.nan
    
    # Align 1-day indicators to lower timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Align 1-week EMA to lower timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(avg_volume_1d_aligned[i]) or
            np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume_1d_aligned[i]
        ema_trend = ema20_1w_aligned[i]
        
        # Alligator values
        jaw_val = jaw_1d_aligned[i]
        teeth_val = teeth_1d_aligned[i]
        lips_val = lips_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (aligned up) + above 1w EMA20 + volume confirmation
            if (lips_val > teeth_val and 
                teeth_val > jaw_val and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Lips < Teeth < Jaw (aligned down) + below 1w EMA20 + volume confirmation
            elif (lips_val < teeth_val and 
                  teeth_val < jaw_val and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator lines cross or price breaks below 1w EMA
            if (lips_val <= teeth_val or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator lines cross or price breaks above 1w EMA
            if (lips_val >= teeth_val or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_WilliamsAlligator_Trend_Volume_v2"
timeframe = "1d"
leverage = 1.0