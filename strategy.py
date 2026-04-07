#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_volume_v2
Hypothesis: On 6-hour timeframe, fade at Camarilla R3/S3 levels from daily pivot and breakout continuation at R4/S4, with volume confirmation.
Long: Price crosses above R3 with volume > 1.5x 20-period average AND close > R3.
Short: Price crosses below S3 with volume > 1.5x 20-period average AND close < S3.
Exit: Price reaches R4/S4 (take profit) or reverses back to R3/S3 (stop).
Uses daily pivot structure for key institutional levels, reducing whipsaw.
Targets 15-25 trades/year to minimize fee drag while capturing meaningful moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_volume_v2"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    pivot = (high + low + close) / 3
    range_ = high - low
    r3 = pivot + range_ * 1.1 / 2
    s3 = pivot - range_ * 1.1 / 2
    r4 = pivot + range_ * 1.1
    s4 = pivot - range_ * 1.1
    return r3, s3, r4, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each daily bar
    r3_vals = np.zeros(len(d_high))
    s3_vals = np.zeros(len(d_high))
    r4_vals = np.zeros(len(d_high))
    s4_vals = np.zeros(len(d_high))
    
    for i in range(len(d_high)):
        r3, s3, r4, s4 = calculate_camarilla(d_high[i], d_low[i], d_close[i])
        r3_vals[i] = r3
        s3_vals[i] = s3
        r4_vals[i] = r4
        s4_vals[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3_vals)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3_vals)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4_vals)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4_vals)
    
    # Volume filter: 6h volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values  # Fill NaN with 1.0 (no volume filter when insufficient data)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from 1 to avoid index issues
        # Skip if Camarilla levels not available
        if np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.5
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Take profit at R4
            if close[i] >= r4_6h[i]:
                exit_long = True
            # Stop loss if price reverses back below R3
            elif close[i] < r3_6h[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Take profit at S4
            if close[i] <= s4_6h[i]:
                exit_short = True
            # Stop loss if price reverses back above S3
            elif close[i] > s3_6h[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price crosses above R3 with volume confirmation
            long_entry = (close[i] > r3_6h[i] and close[i-1] <= r3_6h[i-1]) and vol_confirmed
            
            # Short entry: Price crosses below S3 with volume confirmation
            short_entry = (close[i] < s3_6h[i] and close[i-1] >= s3_6h[i-1]) and vol_confirmed
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals