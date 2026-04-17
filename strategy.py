#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Bollinger Bands (20, 2.0)
    bb_period = 20
    bb_std = 2.0
    
    # Middle band: SMA of close
    sma_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    # Standard deviation
    std_20 = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    # Upper and lower bands
    upper_bb = sma_20 + (bb_std * std_20)
    lower_bb = sma_20 - (bb_std * std_20)
    
    # Bollinger Band Width: (Upper - Lower) / Middle
    bb_width = (upper_bb - lower_bb) / sma_20
    # Handle division by zero
    bb_width = np.where(sma_20 == 0, 0, bb_width)
    
    # Align Bollinger Band Width to 12h timeframe
    bb_width_12h = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Bollinger Band Percentile (20-period) for regime detection
    bb_width_percentile = pd.Series(bb_width_12h).rolling(window=20, min_periods=20).rank(pct=True).values * 100
    
    # Bollinger Band Squeeze detection: low volatility regime
    squeeze_threshold = 20  # Below 20th percentile = squeeze
    bb_squeeze = bb_width_percentile < squeeze_threshold
    
    # Bollinger Band Expansion detection: high volatility regime
    expansion_threshold = 80  # Above 80th percentile = expansion
    bb_expansion = bb_width_percentile > expansion_threshold
    
    # Price position within Bollinger Bands
    # Align BB bands to 12h
    sma_20_12h = align_htf_to_ltf(prices, df_1d, sma_20)
    upper_bb_12h = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_12h = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # %B: (Price - Lower) / (Upper - Lower)
    bb_percent_b = (close - lower_bb_12h) / (upper_bb_12h - lower_bb_12h)
    # Avoid division by zero
    bb_range = upper_bb_12h - lower_bb_12h
    bb_percent_b = np.where(bb_range == 0, 0.5, bb_percent_b)
    
    # Volume filter: current volume > 1.3 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need sufficient data for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bb_percent_b[i]) or np.isnan(volume_ma20[i]) or 
            np.isnan(bb_squeeze[i]) or np.isnan(bb_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        if position == 0:
            # Long entry: price below lower BB (oversold) + BB expansion (volatility increase) + volume
            if bb_percent_b[i] < 0.1 and bb_expansion[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price above upper BB (overbought) + BB expansion + volume
            elif bb_percent_b[i] > 0.9 and bb_expansion[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses above middle band or squeeze ends
            if bb_percent_b[i] > 0.5 or not bb_expansion[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below middle band or squeeze ends
            if bb_percent_b[i] < 0.5 or not bb_expansion[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_BollingerBands_Squeeze_Expansion_Volume"
timeframe = "12h"
leverage = 1.0