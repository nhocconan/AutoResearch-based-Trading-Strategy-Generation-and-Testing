# Hyp: 12h Williams Alligator + volume + RSI filter
# Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs
# Long when Lips > Teeth > Jaw + RSI < 50 + volume > 1.5x avg
# Short when Lips < Teeth < Jaw + RSI > 50 + volume > 1.5x avg
# Exit on Alligator crossover reverse or RSI extreme
# Works in trends (trending markets) and avoids whipsaws with volume filter
# Target: 50-150 trades over 4 years

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def sma(arr, period):
    """Simple moving average with NaN for insufficient data"""
    if period < 1:
        return np.full_like(arr, np.nan)
    result = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        result[i] = np.mean(arr[i - period + 1:i + 1])
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for Alligator
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Williams Alligator lines
    jaw_1d = sma(typical_price_1d, 13)  # 13-period, shifted 8 bars
    teeth_1d = sma(typical_price_1d, 8)   # 8-period, shifted 5 bars
    lips_1d = sma(typical_price_1d, 5)    # 5-period, shifted 3 bars
    
    # Apply shifts (Alligator specific)
    jaw_1d = np.roll(jaw_1d, 8)
    teeth_1d = np.roll(teeth_1d, 5)
    lips_1d = np.roll(lips_1d, 3)
    # Set shifted values to NaN
    jaw_1d[:8] = np.nan
    teeth_1d[:5] = np.nan
    lips_1d[:3] = np.nan
    
    # RSI calculation
    delta = np.diff(close_1d)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/14)
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    for i in range(len(gain)):
        if i < 14:
            continue
        elif i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align all indicators to 12h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need all indicators ready
    start_idx = max(13+8, 8+5, 5+3, 14, vol_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        jaw = jaw_1d_aligned[i]
        teeth = teeth_1d_aligned[i]
        lips = lips_1d_aligned[i]
        rsi = rsi_1d_aligned[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + RSI < 50 + volume confirmation
            if (lips > teeth > jaw and 
                rsi < 50 and 
                vol_ratio > 1.5):
                signals[i] = size
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + RSI > 50 + volume confirmation
            elif (lips < teeth < jaw and 
                  rsi > 50 and 
                  vol_ratio > 1.5):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Alligator bearish cross OR RSI > 70 (overbought)
            if (lips < teeth or 
                rsi > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Alligator bullish cross OR RSI < 30 (oversold)
            if (lips > teeth or 
                rsi < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsAlligator_Volume_RSI"
timeframe = "12h"
leverage = 1.0