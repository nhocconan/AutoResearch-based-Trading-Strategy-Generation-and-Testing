#!/usr/bin/env python3
"""
6h Camarilla Pivot with 1d Trend Filter
Hypothesis: Price reverses at Camarilla pivot levels (S3/R3) on 6h chart when 
daily trend is weak (ADX<25), and breaks out at S4/R4 when daily trend is strong (ADX>25).
Works in bull/bear by adapting to regime: mean reversion in range, breakout in trend.
Target: 15-30 trades/year per symbol.
"""

name = "6h_camarilla_pivot_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for trend filter - call ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 14-period ADX for 1d
    # True Range
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    
    # Directional Movement
    dm_plus_1d = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                          np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus_1d = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                           np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus_1d = np.concatenate([[0], dm_plus_1d])
    dm_minus_1d = np.concatenate([[0], dm_minus_1d])
    
    # Smoothed values
    tr14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    dm_plus_14_1d = pd.Series(dm_plus_1d).rolling(window=14, min_periods=14).sum().values
    dm_minus_14_1d = pd.Series(dm_minus_1d).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus_1d = 100 * dm_plus_14_1d / tr14_1d
    di_minus_1d = 100 * dm_minus_14_1d / tr14_1d
    
    # DX and ADX
    dx_1d = 100 * np.abs(di_plus_1d - di_minus_1d) / (di_plus_1d + di_minus_1d)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(adx_1d[i]):
            signals[i] = 0.0
            continue
        
        # Get aligned 1d ADX for current 6h bar
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)[i]
        
        # Regime detection
        strong_trend_1d = adx_1d_aligned > 25
        
        if position == 1:  # Long position
            # Exit conditions
            if strong_trend_1d:
                # In strong trend: exit on S4 break (stop loss)
                if i >= 2 and low[i] < (close[i-2] + 1.1 * (high[i-2] - low[i-2])):  # S4 approx
                    position = 0
                    signals[i] = 0.0
            else:
                # In weak trend: exit on R3 (take profit)
                if i >= 2 and high[i] > (close[i-2] + 1.0 * (high[i-2] - low[i-2])):  # R3 approx
                    position = 0
                    signals[i] = 0.0
                # Or exit on S3 break (stop loss)
                elif i >= 2 and low[i] < (close[i-2] - 1.0 * (high[i-2] - low[i-2])):  # S3 approx
                    position = 0
                    signals[i] = 0.0
            if position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            if strong_trend_1d:
                # In strong trend: exit on R4 break (stop loss)
                if i >= 2 and high[i] > (close[i-2] - 1.1 * (high[i-2] - low[i-2])):  # R4 approx
                    position = 0
                    signals[i] = 0.0
            else:
                # In weak trend: exit on S3 (take profit)
                if i >= 2 and low[i] < (close[i-2] - 1.0 * (high[i-2] - low[i-2])):  # S3 approx
                    position = 0
                    signals[i] = 0.0
                # Or exit on R3 break (stop loss)
                elif i >= 2 and high[i] > (close[i-2] + 1.0 * (high[i-2] - low[i-2])):  # R3 approx
                    position = 0
                    signals[i] = 0.0
            if position == -1:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if i < 2:
                signals[i] = 0.0
                continue
                
            # Calculate approximate Camarilla levels from 2-period lookback
            # Using high/low from 2 periods ago to avoid look-ahead
            range_val = high[i-2] - low[i-2]
            close_prev = close[i-2]
            
            # Resistance levels
            r3 = close_prev + 1.0 * range_val
            r4 = close_prev + 1.1 * range_val
            # Support levels
            s3 = close_prev - 1.0 * range_val
            s4 = close_prev - 1.1 * range_val
            
            # Entry logic based on regime
            if strong_trend_1d:
                # Strong trend: breakout entries
                if close[i] > r4 and close[i-1] <= r4:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < s4 and close[i-1] >= s4:
                    position = -1
                    signals[i] = -0.25
            else:
                # Weak trend: mean reversion at S3/R3
                if close[i] < s3 and close[i-1] >= s3:
                    position = 1
                    signals[i] = 0.25
                elif close[i] > r3 and close[i-1] <= r3:
                    position = -1
                    signals[i] = -0.25
    
    return signals