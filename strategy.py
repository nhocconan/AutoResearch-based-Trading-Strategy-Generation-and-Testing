#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator system with 1-day trend filter and volume confirmation.
# Uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trend direction and strength.
# Enters long when Lips > Teeth > Jaw (bullish alignment) with price above Jaw and volume confirmation.
# Enters short when Lips < Teeth < Jaw (bearish alignment) with price below Jaw and volume confirmation.
# Includes 1-day EMA50 filter to avoid counter-trend trades and session filter (08-20 UTC).
# Designed for low trade frequency (<30/year) to minimize fee drag while capturing strong trends.
name = "4h_1d_Alligator_EMA50_Volume"
timeframe = "4h"
leverage = 1.0

def _smma(arr, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's smoothing"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    result = np.full_like(arr, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: (prev*(period-1) + current) / period
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA50 trend filter (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator components (SMMA based)
    # Jaw: SMMA(13, 8) - slowest
    jaw = _smma(close, 13)
    # Teeth: SMMA(8, 5) - medium
    teeth = _smma(close, 8)
    # Lips: SMMA(5, 3) - fastest
    lips = _smma(close, 5)
    
    # Volume filter: volume > 1.5 * 30-period average
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_ma[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bullish alignment (Lips > Teeth > Jaw) AND price above Jaw with volume
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > jaw[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment (Lips < Teeth < Jaw) AND price below Jaw with volume
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < jaw[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if alignment breaks or price crosses below Jaw
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or close[i] < jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if alignment breaks or price crosses above Jaw
            if not (lips[i] < teeth[i] and teeth[i] < jaw[i]) or close[i] > jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals