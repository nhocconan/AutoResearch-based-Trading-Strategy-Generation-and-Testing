#!/usr/bin/env python3
"""
1d_1w_RSI_Overbought_Oversold_Simple_v1
Hypothesis: On daily timeframe, use weekly RSI(14) to identify overbought (>70) and oversold (<30) conditions.
Enter short when weekly RSI > 70 and price closes below daily VWAP, enter long when weekly RSI < 30 and price closes above daily VWAP.
Exit when RSI returns to neutral zone (30-70). Works in both bull and bear markets by fading extremes.
Designed for low trade frequency (target 10-25 trades/year) by requiring extreme RSI levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_RSI_Overbought_Oversold_Simple_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for RSI
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 15:  # Need at least 14 periods for RSI
        return np.zeros(n)
    
    # Calculate weekly RSI(14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[14] = np.mean(gain[1:15])  # First average
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1w = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    rsi_1w[:14] = np.nan  # Not enough data
    
    # Align RSI to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate daily VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3
    vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
    # Handle division by zero on first bar
    vwap[volume.cumsum() == 0] = typical_price[volume.cumsum() == 0]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):  # Start after RSI warmup
        # Skip if RSI data is invalid
        if np.isnan(rsi_1w_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Price relative to VWAP
        above_vwap = close[i] > vwap[i]
        below_vwap = close[i] < vwap[i]
        
        # RSI conditions
        rsi_overbought = rsi_1w_aligned[i] > 70
        rsi_oversold = rsi_1w_aligned[i] < 30
        rsi_neutral = (rsi_1w_aligned[i] >= 30) & (rsi_1w_aligned[i] <= 70)
        
        # Entry conditions
        long_entry = rsi_oversold and above_vwap
        short_entry = rsi_overbought and below_vwap
        
        # Exit conditions: RSI returns to neutral
        long_exit = rsi_neutral
        short_exit = rsi_neutral
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals