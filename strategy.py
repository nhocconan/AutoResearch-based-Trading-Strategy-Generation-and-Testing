#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator with 1d momentum filter and volume confirmation.
Long when jaw (13-period SMMA) crosses above teeth (8-period SMMA) with bullish momentum.
Short when jaw crosses below teeth with bearish momentum.
Exit when teeth crosses jaw in opposite direction.
Uses 1d ROC for momentum filter to avoid whipsaws and targets 20-40 trades/year.
Williams Alligator uses smoothed moving averages for smoother trend signals.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def _smma(series, period):
    """Smoothed Moving Average (SMMA) - Wilder's smoothing"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    smma = np.full_like(series, np.nan, dtype=float)
    smma[period-1] = np.mean(series[:period])
    for i in range(period, len(series)):
        smma[i] = (smma[i-1] * (period-1) + series[i]) / period
    return smma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for momentum filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate Williams Alligator (6-period jaw, 5-period teeth, 3-period lips)
    # Using 6, 5, 3 periods for smoother signals on 6h timeframe
    jaw_period = 6   # SMMA of median price
    teeth_period = 5 # SMMA of median price
    
    median_price = (high + low) / 2.0
    jaw = _smma(median_price, jaw_period)
    teeth = _smma(median_price, teeth_period)
    
    # Calculate 1d ROC (10-period) for momentum filter
    close_d = df_daily['close'].values
    roc_d = np.full_like(close_d, np.nan, dtype=float)
    roc_d[10:] = (close_d[10:] - close_d[:-10]) / close_d[:-10] * 100
    
    # Align ROC to 6h timeframe
    roc_aligned = align_htf_to_ltf(prices, df_daily, roc_d)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = np.full_like(volume, np.nan, dtype=float)
    for i in range(20, len(volume)):
        vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(jaw_period, teeth_period, 20), n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(roc_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Jaw crosses above teeth with positive momentum and volume
            if (jaw[i] > teeth[i] and jaw[i-1] <= teeth[i-1] and  # Bullish crossover
                roc_aligned[i] > 0 and                           # Positive momentum
                volume[i] > 1.5 * vol_avg_20[i]):                # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: Jaw crosses below teeth with negative momentum and volume
            elif (jaw[i] < teeth[i] and jaw[i-1] >= teeth[i-1] and  # Bearish crossover
                  roc_aligned[i] < 0 and                           # Negative momentum
                  volume[i] > 1.5 * vol_avg_20[i]):                # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: jaw crosses below teeth (bearish crossover)
                if jaw[i] < teeth[i] and jaw[i-1] >= teeth[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: jaw crosses above teeth (bullish crossover)
                if jaw[i] > teeth[i] and jaw[i-1] <= teeth[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsAlligator_1dROC_Volume"
timeframe = "6h"
leverage = 1.0
#%%