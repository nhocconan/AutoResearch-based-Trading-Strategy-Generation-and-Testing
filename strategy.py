#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly Donchian Breakout with Volume Confirmation and KAMA Trend Filter
# Hypothesis: Weekly Donchian(20) breakouts on 1d chart, filtered by KAMA(10) trend direction and volume spikes,
# capture strong momentum moves while avoiding whipsaws in choppy markets. Weekly trend filter ensures alignment
# with higher timeframe momentum, reducing counter-trend trades. Designed for low frequency (<25/year) to minimize
# fee drag, with discrete position sizing and volatility-based stops.

name = "1d_weekly_donchian_kama_volume_v1"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate KAMA(10) on weekly close
    close_weekly = df_weekly['close'].values
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_weekly, prepend=close_weekly[0]))
    volatility = np.abs(np.diff(close_weekly))
    er = np.zeros_like(close_weekly)
    for i in range(1, len(close_weekly)):
        if np.sum(volatility[max(0, i-9):i+1]) > 0:
            er[i] = change[i] / np.sum(volatility[max(0, i-9):i+1])
        else:
            er[i] = 0
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close_weekly)
    kama[0] = close_weekly[0]
    for i in range(1, len(close_weekly)):
        kama[i] = kama[i-1] + sc[i] * (close_weekly[i] - kama[i-1])
    
    # Align weekly KAMA to daily
    kama_aligned = align_htf_to_ltf(prices, df_weekly, kama)
    
    # Daily Donchian(20)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection: volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(kama_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below weekly KAMA or Donchian lower band
            if close[i] < kama_aligned[i] or close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price crosses above weekly KAMA or Donchian upper band
            if close[i] > kama_aligned[i] or close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian upper band, above weekly KAMA, with volume spike
            if (close[i] > highest_high[i] and 
                close[i] > kama_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian lower band, below weekly KAMA, with volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < kama_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals