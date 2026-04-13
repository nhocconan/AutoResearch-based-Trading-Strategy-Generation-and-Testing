#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator (Jaw/Teeth/Lips) with weekly trend filter and volume confirmation.
# Alligator uses SMAs: Jaw=13, Teeth=8, Lips=5. In trends, lines are ordered (Lips > Teeth > Jaw for uptrend).
# Combined with weekly EMA trend filter and volume spikes, it filters false signals and captures trends.
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
# Works in bull markets (trend following) and bear markets (avoids false signals via weekly filter).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # EMA(20) for weekly trend filter
    ema20_1w = np.zeros(len(close_1w))
    ema_multiplier = 2 / (20 + 1)
    ema20_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema20_1w[i] = (close_1w[i] - ema20_1w[i-1]) * ema_multiplier + ema20_1w[i-1]
    
    # Align weekly EMA to daily timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Williams Alligator on daily timeframe
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
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        weekly_trend = ema20_1w_aligned[i]
        
        # Alligator values
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (aligned up) + above weekly EMA + volume confirmation
            if (lips_val > teeth_val and 
                teeth_val > jaw_val and
                price > weekly_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Lips < Teeth < Jaw (aligned down) + below weekly EMA + volume confirmation
            elif (lips_val < teeth_val and 
                  teeth_val < jaw_val and
                  price < weekly_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Alligator lines cross or price breaks below weekly EMA
            if (lips_val <= teeth_val or
                price < weekly_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Alligator lines cross or price breaks above weekly EMA
            if (lips_val >= teeth_val or
                price > weekly_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_WilliamsAlligator_Trend_Volume"
timeframe = "1d"
leverage = 1.0