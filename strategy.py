#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1-week Relative Strength Index (RSI) divergence and 1-day volume confirmation.
Institutional traders often use weekly RSI divergence to identify trend exhaustion points, which are more reliable on higher timeframes.
We enter long when bullish divergence occurs (price makes lower low, RSI makes higher low) with above-average volume,
and enter short when bearish divergence occurs (price makes higher high, RSI makes lower high) with above-average volume.
Exits when RSI returns to neutral territory (40-60 range) or opposite divergence occurs.
Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.
Works in both bull and bear markets as divergence signals reversals regardless of trend direction.
"""

name = "6h_WeeklyRSI_Divergence_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-week RSI (14-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close']
    delta = close_1w.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Wilder's smoothing
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_1w = 100 - (100 / (1 + rs))
    
    rsi_1w_values = rsi_1w.values
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w_values)
    
    # Calculate 1-day average volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume']
    avg_volume_1d = volume_1d.rolling(window=20, min_periods=20).mean()
    volume_1d_values = avg_volume_1d.values
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_values)
    
    # Current 6h volume
    volume_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Detect RSI divergence - need to find peaks and troughs
    # Bullish divergence: price makes lower low, RSI makes higher low
    # Bearish divergence: price makes higher high, RSI makes lower high
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(volume_1d_aligned[i]) or 
            np.isnan(volume_ma_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for bullish divergence: price lower low, RSI higher low
            # Simple approximation: check if RSI is oversold and turning up while price is near lows
            bullish_divergence = (
                rsi_1w_aligned[i] < 30 and  # Oversold
                rsi_1w_aligned[i] > rsi_1w_aligned[i-1] and  # RSI rising
                close[i] <= low[i-1]  # Price at or below recent low
            )
            
            # Look for bearish divergence: price higher high, RSI lower high
            bearish_divergence = (
                rsi_1w_aligned[i] > 70 and  # Overbought
                rsi_1w_aligned[i] < rsi_1w_aligned[i-1] and  # RSI falling
                close[i] >= high[i-1]  # Price at or above recent high
            )
            
            # Volume confirmation: current volume above daily average
            volume_confirmed = volume[i] > volume_1d_aligned[i]
            
            if bullish_divergence and volume_confirmed:
                signals[i] = 0.25
                position = 1
            elif bearish_divergence and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral or bearish divergence
            exit_condition = (
                rsi_1w_aligned[i] >= 40 or  # RSI back to neutral
                (rsi_1w_aligned[i] > 70 and rsi_1w_aligned[i] < rsi_1w_aligned[i-1])  # Bearish divergence
            )
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral or bullish divergence
            exit_condition = (
                rsi_1w_aligned[i] <= 60 or  # RSI back to neutral
                (rsi_1w_aligned[i] < 30 and rsi_1w_aligned[i] > rsi_1w_aligned[i-1])  # Bullish divergence
            )
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals