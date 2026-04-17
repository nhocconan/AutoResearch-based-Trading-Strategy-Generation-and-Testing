#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Bollinger Bands (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Bollinger Bands (20-period, 2.5 std)
    period = 20
    mult = 2.5
    
    # Bollinger Bands calculation
    sma = pd.Series(close_1d).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close_1d).rolling(window=period, min_periods=period).std().values
    
    upper = sma + mult * std
    lower = sma - mult * std
    
    # Bandwidth for squeeze detection
    bandwidth = (upper - lower) / sma
    
    # Align Bollinger Bands to 12h
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    bandwidth_aligned = align_htf_to_ltf(prices, df_1d, bandwidth)
    
    # 12h Bollinger Bands for entry/exit
    sma_12h = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_12h = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_12h = sma_12h + 2.0 * std_12h
    lower_12h = sma_12h - 2.0 * std_12h
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # Need daily and 12h BBands
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(bandwidth_aligned[i]) or
            np.isnan(upper_12h[i]) or 
            np.isnan(lower_12h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: daily bandwidth < 0.05 (low volatility squeeze)
        vol_filter = bandwidth_aligned[i] < 0.05
        
        # Volume confirmation
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long: price breaks above 12h upper BB during low vol squeeze with volume
            if (close[i] > upper_12h[i] and vol_filter and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h lower BB during low vol squeeze with volume
            elif (close[i] < lower_12h[i] and vol_filter and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to 12h middle band
            if close[i] < sma_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to 12h middle band
            if close[i] > sma_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_BollingerSqueeze_Breakout_Volume"
timeframe = "12h"
leverage = 1.0