#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1-day trend filter and volume confirmation.
# Long when price above Alligator's Jaw (13-period SMMA) with 1d EMA50 uptrend and volume > 1.5x average.
# Short when price below Jaw with 1d EMA50 downtrend and volume > 1.5x average.
# Exit when price crosses the Teeth (8-period SMMA).
# Uses Alligator for trend identification, 1d EMA50 for higher timeframe trend filter, volume for confirmation.
# Target: 12-37 trades/year to avoid fee drift. Works in bull/bear via trend-aligned entries.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator SMMA components
    jaw_period = 13   # Jaw (Blue)
    teeth_period = 8  # Teeth (Red)
    lips_period = 5   # Lips (Green)
    
    def smma(arr, period):
        """Smoothed Moving Average (SMMA)"""
        result = np.full(len(arr), np.nan)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period - 1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period - 1) + arr[i]) / period
        return result
    
    jaw = smma(close, jaw_period)
    teeth = smma(close, teeth_period)
    lips = smma(close, lips_period)
    
    # Calculate 1-day EMA50 for trend filter
    ema_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Get volume MA for confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    # Align 1-day indicators to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Alligator components, EMA50, and volume MA20
    start_idx = max(jaw_period - 1, teeth_period - 1, lips_period - 1, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price above Jaw with 1d EMA50 uptrend and volume
            if (price > jaw[i] and 
                price > ema_1d_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: price below Jaw with 1d EMA50 downtrend and volume
            elif (price < jaw[i] and 
                  price < ema_1d_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Teeth
            if price < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Teeth
            if price > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsAlligator_Jaw_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0