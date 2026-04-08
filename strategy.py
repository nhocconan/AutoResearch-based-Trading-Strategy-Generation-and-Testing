#!/usr/bin/env python3
"""
6h_1d_rsi_momentum_v1
Hypothesis: Use 1d RSI to determine trend direction, 6h RSI for entry timing with volume confirmation.
Long when 1d RSI > 50 (bullish) and 6h RSI crosses above 30 from below with volume surge.
Short when 1d RSI < 50 (bearish) and 6h RSI crosses below 70 from above with volume surge.
Designed for low trade frequency (10-25/year) to minimize fee drift.
Works in bull/bear via 1d trend filter and mean-reversion entries within trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_rsi_momentum_v1"
timeframe = "6h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Calculate RSI with Wilder's smoothing"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan, dtype=float)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    rsi_1d = calculate_rsi(close_1d, 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 6h RSI for entry timing
    rsi_6h = calculate_rsi(close, 14)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if np.isnan(rsi_1d_aligned[i]) or np.isnan(rsi_6h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        rsi_1d_val = rsi_1d_aligned[i]
        rsi_6h_val = rsi_6h[i]
        
        if position == 1:  # Long
            # Exit: 6h RSI crosses below 50 or volume drops
            if rsi_6h_val < 50 or vol_ratio < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: 6h RSI crosses above 50 or volume drops
            if rsi_6h_val > 50 or vol_ratio < 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: 1d bullish trend + 6h RSI crosses above 30 from below with volume
            if (rsi_1d_val > 50 and 
                rsi_6h_val > 30 and 
                rsi_6h[i-1] <= 30 and 
                vol_ratio > 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: 1d bearish trend + 6h RSI crosses below 70 from above with volume
            elif (rsi_1d_val < 50 and 
                  rsi_6h_val < 70 and 
                  rsi_6h[i-1] >= 70 and 
                  vol_ratio > 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals