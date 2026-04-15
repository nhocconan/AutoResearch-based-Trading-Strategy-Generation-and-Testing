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
    
    # Get daily data for context (weekly pivot uses weekly high/low/close)
    weekly = get_htf_data(prices, '1w')
    weekly_high = weekly['high'].values
    weekly_low = weekly['low'].values
    weekly_close = weekly['close'].values
    
    # Calculate weekly pivot points
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    
    # Align weekly levels to 12h timeframe (wait for weekly close)
    weekly_pivot_12h = align_htf_to_ltf(prices, weekly, weekly_pivot)
    weekly_r1_12h = align_htf_to_ltf(prices, weekly, weekly_r1)
    weekly_s1_12h = align_htf_to_ltf(prices, weekly, weekly_s1)
    weekly_r2_12h = align_htf_to_ltf(prices, weekly, weekly_r2)
    weekly_s2_12h = align_htf_to_ltf(prices, weekly, weekly_s2)
    
    # Volume filter: current volume > 1.5x 50-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Range filter: avoid trading too close to pivot (±0.5%)
    price_to_pivot = np.abs(close - weekly_pivot_12h) / weekly_pivot_12h
    range_filter = price_to_pivot > 0.005
    
    # Volatility filter: avoid low volatility regimes
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr / close
    volatility_filter = atr_ratio > 0.01  # Adjusted for 12h
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_12h[i]) or np.isnan(weekly_r1_12h[i]) or 
            np.isnan(weekly_s1_12h[i]) or np.isnan(weekly_r2_12h[i]) or 
            np.isnan(weekly_s2_12h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when all filters pass
        if volume_filter[i] and range_filter[i] and volatility_filter[i]:
            # Long: break above R2 with volume
            if close[i] > weekly_r2_12h[i]:
                signals[i] = 0.25
            # Short: break below S2 with volume
            elif close[i] < weekly_s2_12h[i]:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_WeeklyPivot_R2_S2_Breakout_Volume_RangeVolatilityFilter"
timeframe = "12h"
leverage = 1.0