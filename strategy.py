#!/usr/bin/env python3
"""
Hypothesis: Daily Williams Alligator with 1-week trend filter and volume confirmation.
Long when price > Alligator's Jaw (SMMA13) with weekly EMA20 rising and volume spike.
Short when price < Alligator's Jaw with weekly EMA20 falling and volume spike.
Exit when price crosses Alligator's Teeth (SMMA8).
Designed for low trade frequency by requiring trend alignment and volume confirmation.
Works in both bull and bear markets by following the weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(series, period):
    """Smoothed Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    smma_vals = np.full_like(series, np.nan, dtype=float)
    smma_vals[period-1] = np.mean(series[:period])
    for i in range(period, len(series)):
        smma_vals[i] = (smma_vals[i-1] * (period-1) + series[i]) / period
    return smma_vals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Williams Alligator - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams Alligator components using daily data
    median_price = (df_1d['high'].values + df_1d['low'].values) / 2.0
    jaw = smma(median_price, 13)   # Blue line
    teeth = smma(median_price, 8)  # Red line
    lips = smma(median_price, 5)   # Green line
    
    # Align to daily timeframe (each day's values apply to the entire day)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after enough data for Alligator
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price > Jaw with weekly EMA20 rising and volume spike
            if (close[i] > jaw_aligned[i] and 
                ema20_1w_aligned[i] > ema20_1w_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price < Jaw with weekly EMA20 falling and volume spike
            elif (close[i] < jaw_aligned[i] and 
                  ema20_1w_aligned[i] < ema20_1w_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses Teeth
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below Teeth
                if close[i] < teeth_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above Teeth
                if close[i] > teeth_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Williams_Alligator_1wEMA20_Trend_Volume"
timeframe = "1d"
leverage = 1.0