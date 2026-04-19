#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1-day Exponential Moving Average (EMA) cross and 
# Bollinger Bands width regime filter. Long when price > EMA50 and BB width expanding 
# (volatility increasing), short when price < EMA50 and BB width contracting. 
# Uses volume confirmation to avoid false breakouts. Works in trending markets 
# by capturing momentum with volatility filters. Target: 50-150 total trades over 4 years.
name = "6h_EMA50_BBWidth_Volume_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA and Bollinger Bands (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily close
    close_series = pd.Series(close_1d)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Bollinger Bands (20, 2) on daily close
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Align daily indicators to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Volume filter: volume > 1.5 * 30-period average (more conservative)
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA and BB calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(bb_width_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above EMA50 AND BB width increasing (volatility expansion) 
            if close[i] > ema_50_aligned[i] and bb_width_aligned[i] > bb_width_aligned[i-1] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below EMA50 AND BB width decreasing (volatility contraction)
            elif close[i] < ema_50_aligned[i] and bb_width_aligned[i] < bb_width_aligned[i-1] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price crosses below EMA50 or BB width contracts significantly
            if close[i] <= ema_50_aligned[i] or bb_width_aligned[i] < bb_width_aligned[i-20]:  # Significant contraction
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price crosses above EMA50 or BB width expands significantly
            if close[i] >= ema_50_aligned[i] or bb_width_aligned[i] > bb_width_aligned[i-20]:  # Significant expansion
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals