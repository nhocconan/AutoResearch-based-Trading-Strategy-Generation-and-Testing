#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R (14) with 1-day trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; extreme readings (>80 or <20)
# combined with 1-day trend direction (price above/below 50 EMA) and volume spike
# provide high-probability mean-reversion entries. Works in both bull and bear markets
# by fading extremes only when aligned with higher timeframe trend.
# Target: 80-150 total trades over 4 years (20-38/year) to balance opportunity and cost.
name = "4h_WilliamsR_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def williams_r(high, low, close, period=14):
    """Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return wr.values

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R on 4h timeframe
    wr = williams_r(high, low, close, 14)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 1.8 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(wr[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
            
        # Long setup: Williams %R oversold (< -80), price above 1d EMA50, volume confirmation
        long_setup = (wr[i] < -80) and (close[i] > ema50_1d_aligned[i]) and volume_filter[i]
        
        # Short setup: Williams %R overbought (> -20), price below 1d EMA50, volume confirmation
        short_setup = (wr[i] > -20) and (close[i] < ema50_1d_aligned[i]) and volume_filter[i]
        
        if position == 0:
            # Enter long on oversold with trend and volume
            if long_setup:
                signals[i] = 0.25
                position = 1
            # Enter short on overbought with trend and volume
            elif short_setup:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: Williams %R returns to neutral (> -50) or trend breaks
            if wr[i] > -50 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: Williams %R returns to neutral (< -50) or trend breaks
            if wr[i] < -50 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals