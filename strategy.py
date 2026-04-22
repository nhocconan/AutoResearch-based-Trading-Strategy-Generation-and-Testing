#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams Alligator with 1-day EMA trend and volume confirmation.
Long when price is above Alligator teeth (green line) with 1-day EMA50 rising and volume spike.
Short when price is below Alligator teeth with 1-day EMA50 falling and volume spike.
Exit when price crosses Alligator lips (red line).
Williams Alligator identifies trend presence and direction; 1-day EMA50 filters for higher timeframe trend;
volume spike confirms institutional participation. Designed for low trade frequency by requiring
multiple confirmations. Works in both bull and bear markets by following the daily trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA)"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    result = np.full_like(arr, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: (prev_smma * (period-1) + current_price) / period
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1-day EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator on 12h timeframe
    # Jaw (blue line): SMMA(median, 13) shifted 8 bars forward
    # Teeth (red line): SMMA(median, 8) shifted 5 bars forward  
    # Lips (green line): SMMA(median, 5) shifted 3 bars forward
    median_price = (high + low + close) / 3.0
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Shift the lines as per Alligator specification
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted positions that would look ahead
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough data for EMA50 and Alligator
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(lips_shifted[i]) or np.isnan(teeth_shifted[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price above teeth (red line) with 1-day EMA50 rising and volume spike
            if (close[i] > teeth_shifted[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price below teeth (red line) with 1-day EMA50 falling and volume spike
            elif (close[i] < teeth_shifted[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses lips (green line)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below lips
                if close[i] < lips_shifted[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above lips
                if close[i] > lips_shifted[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Williams_Alligator_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0