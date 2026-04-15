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
    
    # Get weekly data for structure
    weekly = get_htf_data(prices, '1w')
    weekly_high = weekly['high'].values
    weekly_low = weekly['low'].values
    weekly_close = weekly['close'].values
    
    # Get daily data for context
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    daily_volume = daily['volume'].values
    
    # Calculate weekly ATR for volatility regime
    weekly_close_prev = np.concatenate([[weekly_close[0]], weekly_close[:-1]])
    tr_weekly = np.maximum(weekly_high - weekly_low,
                           np.maximum(np.abs(weekly_high - weekly_close_prev),
                                      np.abs(weekly_low - weekly_close_prev)))
    atr_weekly = pd.Series(tr_weekly).rolling(window=14, min_periods=14).mean().values
    atr_ratio_weekly = atr_weekly / weekly_close
    
    # Align weekly ATR ratio to 6h timeframe
    atr_ratio_6h = align_htf_to_ltf(prices, weekly, atr_ratio_weekly)
    
    # Calculate weekly pivot points (standard calculation)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_6h = align_htf_to_ltf(prices, weekly, weekly_pivot)
    weekly_r1_6h = align_htf_to_ltf(prices, weekly, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, weekly, weekly_s1)
    weekly_r2_6h = align_htf_to_ltf(prices, weekly, weekly_r2)
    weekly_s2_6h = align_htf_to_ltf(prices, weekly, weekly_s2)
    weekly_r3_6h = align_htf_to_ltf(prices, weekly, weekly_r3)
    weekly_s3_6h = align_htf_to_ltf(prices, weekly, weekly_s3)
    
    # Calculate daily volume ratio for confirmation
    daily_vol_ma = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    daily_vol_ratio = daily_volume / (daily_vol_ma + 1e-10)
    daily_vol_ratio_6h = align_htf_to_ltf(prices, daily, daily_vol_ratio)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_ratio_6h[i]) or np.isnan(weekly_pivot_6h[i]) or 
            np.isnan(weekly_r1_6h[i]) or np.isnan(weekly_s1_6h[i]) or
            np.isnan(daily_vol_ratio_6h[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in high volatility regimes (avoid choppy markets)
        if atr_ratio_6h[i] < 0.015:
            signals[i] = 0.0
            continue
        
        # Require volume confirmation
        if daily_vol_ratio_6h[i] < 1.2:
            signals[i] = 0.0
            continue
        
        # Long setup: price breaks above weekly R1 with volume
        if (close[i] > weekly_r1_6h[i] and 
            close[i-1] <= weekly_r1_6h[i-1]):
            signals[i] = 0.25
        # Short setup: price breaks below weekly S1 with volume
        elif (close[i] < weekly_s1_6h[i] and 
              close[i-1] >= weekly_s1_6h[i-1]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_Breakout_Volume_Filter"
timeframe = "6h"
leverage = 1.0