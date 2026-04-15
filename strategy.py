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
    
    # Get daily data for key levels
    daily = get_htf_data(prices, '1d')
    
    # Calculate daily high and low
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    
    # Align daily levels to current timeframe
    daily_high_aligned = align_htf_to_ltf(prices, daily, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, daily, daily_low)
    
    # Volume spike detection: current volume > 1.5x 20-day average volume
    vol_ma_20d = pd.Series(daily['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20d_aligned = align_htf_to_ltf(prices, daily, vol_ma_20d)
    volume_threshold = 1.5 * vol_ma_20d_aligned
    volume_spike = volume > volume_threshold
    
    # Volatility filter: average daily range > 1% of price
    daily_range = daily_high - daily_low
    avg_daily_range = pd.Series(daily_range).rolling(window=20, min_periods=20).mean().values
    avg_daily_range_aligned = align_htf_to_ltf(prices, daily, avg_daily_range)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if np.isnan(daily_high_aligned[i]) or np.isnan(daily_low_aligned[i]) or \
           np.isnan(vol_ma_20d_aligned[i]) or np.isnan(avg_daily_range_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Only trade when volatility is sufficient (avoid low volatility chop)
        if avg_daily_range_aligned[i] < 0.01 * close[i]:  # Less than 1% of price
            signals[i] = 0.0
            continue
            
        # Long: Price breaks above daily high with volume spike
        if close[i] > daily_high_aligned[i] and volume_spike[i]:
            signals[i] = 0.25
        
        # Short: Price breaks below daily low with volume spike
        elif close[i] < daily_low_aligned[i] and volume_spike[i]:
            signals[i] = -0.25
        
        # Exit: reverse signal when price returns to opposite daily level
        elif close[i] < daily_low_aligned[i] and signals[i-1] > 0:
            signals[i] = 0.0
        elif close[i] > daily_high_aligned[i] and signals[i-1] < 0:
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_1D_Range_Breakout_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0