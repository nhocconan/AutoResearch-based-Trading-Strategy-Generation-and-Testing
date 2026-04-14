#!/usr/bin/env python3
"""
Hypothesis: 6h momentum with weekly exhaustion detection. Uses weekly RSI extremes
combined with 6h price action to catch reversals in both bull and bear markets.
Goes long when weekly RSI < 30 and 6h closes above prior high with volume.
Goes short when weekly RSI > 70 and 6h closes below prior low with volume.
Exits when weekly RSI returns to neutral (40-60) or opposite extreme forms.
Designed for low turnover: ~20-40 trades/year per symbol.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly RSI(14)
    rsi_period = 14
    delta = pd.Series(df_1w['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # 6h prior bar high/low for breakout confirmation
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    high_prev[0] = high[0]
    low_prev[0] = low[0]
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(20, n):
        # Weekly index (6 bars per week: 7*24/6 = 28, but use 28 for exact)
        idx_1w = i // 28
        if idx_1w < rsi_period:
            continue
        
        # Get previous weekly RSI to avoid look-ahead
        rsi_prev = rsi_values[idx_1w - 1] if idx_1w - 1 < len(rsi_values) else rsi_values[-1]
        if np.isnan(rsi_prev):
            continue
        
        # Create array for alignment (using previous value)
        rsi_arr = np.full(len(df_1w), rsi_prev)
        rsi_6h = align_htf_to_ltf(prices, df_1w, rsi_arr)[i]
        
        if position == 0:
            # Long: weekly oversold + 6h breaks above prior high + volume
            if (rsi_6h < 30 and 
                close[i] > high_prev[i] and 
                volume[i] > vol_ma[i] * 1.5):
                position = 1
                signals[i] = position_size
            # Short: weekly overbought + 6h breaks below prior low + volume
            elif (rsi_6h > 70 and 
                  close[i] < low_prev[i] and 
                  volume[i] > vol_ma[i] * 1.5):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: weekly RSI returns to neutral or overbought
            if rsi_6h >= 40:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: weekly RSI returns to neutral or oversold
            if rsi_6h <= 60:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyRSI_Exhaustion"
timeframe = "6h"
leverage = 1.0