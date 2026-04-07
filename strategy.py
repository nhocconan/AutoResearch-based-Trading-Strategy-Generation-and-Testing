#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Daily KAMA with Weekly Trend Filter and Volume Spike
# Hypothesis: KAMA adapts to market noise, providing reliable trend signals.
# In bull markets: price above rising KAMA = long. In bear markets: price below falling KAMA = short.
# Weekly trend filter (1w KAMA) ensures alignment with higher timeframe trend.
# Volume spike (2x 20-period average) confirms institutional participation.
# Target: 15-30 trades/year (60-120 over 4 years).

name = "1d_kama_weekly_trend_volume_v1"
timeframe = "1d"
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
    
    # Calculate KAMA on daily data
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_s - close_s.shift(10))
    volatility = abs(close_s.diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility
    er = er.fillna(0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = [close[0]]  # Initialize with first close
    for i in range(1, n):
        kama.append(kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1]))
    kama = np.array(kama)
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly KAMA
    weekly_close = df_weekly['close'].values
    weekly_close_s = pd.Series(weekly_close)
    weekly_change = abs(weekly_close_s - weekly_close_s.shift(10))
    weekly_volatility = abs(weekly_close_s.diff()).rolling(window=10, min_periods=10).sum()
    weekly_er = weekly_change / weekly_volatility
    weekly_er = weekly_er.fillna(0)
    weekly_sc = (weekly_er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    weekly_kama = [weekly_close[0]]
    for i in range(1, len(weekly_close)):
        weekly_kama.append(weekly_kama[i-1] + weekly_sc.iloc[i] * (weekly_close[i] - weekly_kama[i-1]))
    weekly_kama = np.array(weekly_kama)
    
    # Align weekly KAMA to daily
    weekly_kama_aligned = align_htf_to_ltf(prices, df_weekly, weekly_kama)
    
    # Volume filter: volume > 2x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(weekly_kama_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below KAMA or weekly trend turns bearish
            if (close[i] < kama[i] or weekly_kama_aligned[i] > weekly_kama_aligned[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above KAMA or weekly trend turns bullish
            if (close[i] > kama[i] or weekly_kama_aligned[i] < weekly_kama_aligned[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price above daily KAMA, weekly KAMA rising, volume spike
            if (close[i] > kama[i] and 
                weekly_kama_aligned[i] > weekly_kama_aligned[i-1] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price below daily KAMA, weekly KAMA falling, volume spike
            elif (close[i] < kama[i] and 
                  weekly_kama_aligned[i] < weekly_kama_aligned[i-1] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals