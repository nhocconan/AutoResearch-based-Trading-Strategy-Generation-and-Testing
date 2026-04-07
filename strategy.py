#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Weekly KAMA Trend with Volume Filter
# Hypothesis: Weekly KAMA (Kaufman Adaptive Moving Average) adapts to market noise, 
# providing a reliable trend filter that works in both bull and bear markets. 
# Price above weekly KAMA indicates bullish trend, below indicates bearish trend. 
# Volume confirmation ensures institutional participation. 
# Uses tight entry conditions to limit trades and avoid fee drag.
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_weekly_kama_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for KAMA calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate weekly KAMA (Kaufman Adaptive Moving Average)
    weekly_close = df_weekly['close'].values
    
    # Efficiency Ratio (ER) and Smoothing Constants
    change = np.abs(np.diff(weekly_close, prepend=weekly_close[0]))
    volatility = np.abs(np.diff(weekly_close))
    er = np.zeros_like(weekly_close)
    for i in range(1, len(weekly_close)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 1.0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(weekly_close)
    kama[0] = weekly_close[0]
    for i in range(1, len(weekly_close)):
        kama[i] = kama[i-1] + sc[i] * (weekly_close[i] - kama[i-1])
    
    # Shift by 1 to use previous week's KAMA (avoid look-ahead)
    prev_weekly_kama = np.roll(kama, 1)
    prev_weekly_kama[0] = prev_weekly_kama[1] if len(prev_weekly_kama) > 1 else weekly_close[0]
    
    # Align to 4h timeframe (use previous week's KAMA)
    weekly_kama_aligned = align_htf_to_ltf(prices, df_weekly, prev_weekly_kama)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(weekly_kama_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below weekly KAMA or volume drops
            if close[i] <= weekly_kama_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price crosses above weekly KAMA or volume drops
            if close[i] >= weekly_kama_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price crosses above weekly KAMA with volume
            if close[i] > weekly_kama_aligned[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: price crosses below weekly KAMA with volume
            elif close[i] < weekly_kama_aligned[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals