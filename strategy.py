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
    
    # Get daily data for pivot points and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (classic)
    daily_pivot = (high_1d + low_1d + close_1d) / 3.0
    daily_r1 = 2 * daily_pivot - low_1d
    daily_s1 = 2 * daily_pivot - high_1d
    
    # Align daily pivot levels to 4h timeframe
    daily_pivot_4h = align_htf_to_ltf(prices, df_1d, daily_pivot)
    daily_r1_4h = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_4h = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    # Calculate daily ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d[0:13] = np.nan  # Ensure proper warmup
    atr_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need daily ATR, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(daily_pivot_4h[i]) or 
            np.isnan(daily_r1_4h[i]) or 
            np.isnan(daily_s1_4h[i]) or 
            np.isnan(atr_4h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Volatility filter: ATR > 50% of its 50-period average
        atr_ma50 = pd.Series(atr_4h).rolling(window=50, min_periods=50).mean()
        atr_ratio = atr_4h[i] / atr_ma50.iloc[i] if not np.isnan(atr_ma50.iloc[i]) else 0
        volatility_filter = atr_ratio > 0.5
        
        # Price relative to daily pivot levels
        price_above_r1 = close[i] > daily_r1_4h[i]
        price_below_s1 = close[i] < daily_s1_4h[i]
        
        if position == 0:
            # Long: Price breaks above daily R1 with volume and volatility
            if (price_above_r1 and volume_filter and volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily S1 with volume and volatility
            elif (price_below_s1 and volume_filter and volatility_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below daily pivot
            if close[i] < daily_pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above daily pivot
            if close[i] > daily_pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DailyPivot_Breakout_Vol_VolFilter"
timeframe = "4h"
leverage = 1.0