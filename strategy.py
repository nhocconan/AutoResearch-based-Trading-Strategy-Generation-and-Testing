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
    
    # Get weekly data for trend and key levels
    weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly high and low for breakout levels
    weekly_high = weekly['high'].values
    weekly_low = weekly['low'].values
    
    # Weekly high and low aligned to daily
    weekly_high_aligned = align_htf_to_ltf(prices, weekly, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, weekly, weekly_low)
    
    # Volume spike detection: current volume > 2x 20-day average volume
    # Calculate daily volume first, then get 20-day average
    daily = get_htf_data(prices, '1d')
    daily_vol = daily['volume'].values
    vol_ma_20d = pd.Series(daily_vol).rolling(window=20, min_periods=20).mean().values
    vol_ma_20d_aligned = align_htf_to_ltf(prices, daily, vol_ma_20d)
    volume_threshold = 2.0 * vol_ma_20d_aligned
    volume_spike = volume > volume_threshold
    
    # Weekly trend filter: price above/below weekly EMA20
    weekly_close = weekly['close'].values
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema20_aligned = align_htf_to_ltf(prices, weekly, weekly_ema20)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(weekly_ema20_aligned[i]) or np.isnan(vol_ma_20d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long: Price breaks above weekly high with volume spike and above weekly EMA20
        if (close[i] > weekly_high_aligned[i] and volume_spike[i] and 
            close[i] > weekly_ema20_aligned[i]):
            signals[i] = 0.25
        
        # Short: Price breaks below weekly low with volume spike and below weekly EMA20
        elif (close[i] < weekly_low_aligned[i] and volume_spike[i] and 
              close[i] < weekly_ema20_aligned[i]):
            signals[i] = -0.25
        
        # Exit: reverse signal when price returns to opposite weekly level
        elif close[i] < weekly_low_aligned[i] and signals[i-1] > 0:
            signals[i] = 0.0
        elif close[i] > weekly_high_aligned[i] and signals[i-1] < 0:
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_WeeklyHighLow_Breakout_Volume_Trend_Filter"
timeframe = "1d"
leverage = 1.0