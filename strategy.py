#!/usr/bin/env python3
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h Bollinger Bands (20, 2) for regime detection
    sma_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).mean()
    std_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).std()
    upper_12h = sma_12h + 2 * std_12h
    lower_12h = sma_12h - 2 * std_12h
    
    # Align 12h Bollinger Bands to 6h
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h.values)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h.values)
    
    # 12h ADX for trend strength (14 period)
    # Calculate +DM, -DM, TR
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    tr = np.maximum(high_12h[1:] - low_12h[1:], 
                    np.maximum(abs(high_12h[1:] - high_12h[:-1]), 
                               abs(low_12h[1:] - low_12h[:-1])))
    
    # Pad arrays to match length
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    tr = np.concatenate([[0], tr])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr_12h = np.zeros_like(tr)
    atr_12h[0] = tr[0]
    for i in range(1, len(tr)):
        atr_12h[i] = atr_12h[i-1] + (tr[i] - atr_12h[i-1]) / 14
    
    di_plus = np.zeros_like(tr)
    di_minus = np.zeros_like(tr)
    for i in range(14, len(tr)):
        if atr_12h[i] != 0:
            di_plus[i] = 100 * np.mean(dm_plus[i-13:i+1]) / atr_12h[i]
            di_minus[i] = 100 * np.mean(dm_minus[i-13:i+1]) / atr_12h[i]
        else:
            di_plus[i] = 0
            di_minus[i] = 0
    
    dx = np.zeros_like(tr)
    for i in range(14, len(tr)):
        if di_plus[i] + di_minus[i] != 0:
            dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
        else:
            dx[i] = 0
    
    adx_12h = np.zeros_like(dx)
    adx_12h[0] = dx[0]
    for i in range(1, len(dx)):
        adx_12h[i] = adx_12h[i-1] + (dx[i] - adx_12h[i-1]) / 14
    
    # Align 12h ADX to 6h
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 6h Bollinger Bands (20, 2) for entry signals
    sma_6h = pd.Series(close).rolling(window=20, min_periods=20).mean()
    std_6h = pd.Series(close).rolling(window=20, min_periods=20).std()
    upper_6h = sma_6h + 2 * std_6h
    lower_6h = sma_6h - 2 * std_6h
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_6h[i]) or np.isnan(lower_6h[i]) or 
            np.isnan(vol_threshold[i]) or np.isnan(adx_12h_aligned[i]) or
            np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i])):
            continue
        
        # Only trade in strong trends (ADX > 25)
        if adx_12h_aligned[i] <= 25:
            signals[i] = 0.0
            continue
        
        # Long: price above 12h upper band AND breaks above 6h upper band + volume
        if (close[i] > upper_12h_aligned[i] and 
            close[i] > upper_6h[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price below 12h lower band AND breaks below 6h lower band + volume
        elif (close[i] < lower_12h_aligned[i] and 
              close[i] < lower_6h[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price crosses back inside 6h bands (mean reversion within trend)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < upper_6h[i]) or
               (signals[i-1] == -0.25 and close[i] > lower_6h[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_12h_ADX_Bollinger_Breakout"
timeframe = "6h"
leverage = 1.0