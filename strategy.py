#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Load 1h data for volatility calculation
    df_1h = get_htf_data(prices, '1h')
    atr_1h = calculate_atr(df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, 14)
    atr_1h_aligned = align_htf_to_ltf(prices, df_1h, atr_1h)
    
    # Calculate ATR-based volatility threshold
    vol_threshold = atr_1h_aligned * 1.5
    
    # Price range for the last 24 hours (2 periods on 12h)
    high_24h = pd.Series(prices['high']).rolling(window=2, min_periods=2).max()
    low_24h = pd.Series(prices['low']).rolling(window=2, min_periods=2).min()
    range_24h = high_24h - low_24h
    
    signals = np.zeros(n)
    
    for i in range(2, n):
        # Skip if any required data is NaN
        if np.isnan(range_24h[i]) or np.isnan(vol_threshold[i]):
            continue
        
        # Long: price breaks above 24h high with sufficient volatility
        if prices['high'][i] > high_24h[i] and range_24h[i] > vol_threshold[i]:
            signals[i] = 0.25
        
        # Short: price breaks below 24h low with sufficient volatility
        elif prices['low'][i] < low_24h[i] and range_24h[i] > vol_threshold[i]:
            signals[i] = -0.25
        
        # Exit: price returns to the middle of the 24h range
        elif i > 0 and signals[i-1] != 0:
            mid_point = (high_24h[i] + low_24h[i]) / 2
            if (signals[i-1] == 0.25 and prices['close'][i] < mid_point) or \
               (signals[i-1] == -0.25 and prices['close'][i] > mid_point):
                signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

def calculate_atr(high, low, close, period):
    """Calculate Average True Range"""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    return atr.values

name = "12h_Volatility_Breakout"
timeframe = "12h"
leverage = 1.0