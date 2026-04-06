#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation
# Enter long when: price breaks above 20-day high, weekly trend is bullish (price > weekly EMA50), volume > 1.5x average
# Enter short when: price breaks below 20-day low, weekly trend is bearish (price < weekly EMA50), volume > 1.5x average
# Uses weekly trend to filter counter-trend breakouts, volume to confirm breakout strength
# Exit when price crosses 10-day EMA in opposite direction or opposite breakout occurs
# Target: 50-100 trades over 4 years by combining trend filter with breakout logic

name = "1d_donchian20_weekly_trend_vol_v1"
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
    
    # Weekly trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    weekly_ema50 = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 10-day EMA for exit
    ema10 = pd.Series(close).ewm(span=10, adjust=False).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(weekly_ema50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below 10-day EMA OR short breakout occurs
            if close[i] < ema10[i] or close[i] < low_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above 10-day EMA OR long breakout occurs
            if close[i] > ema10[i] or close[i] > high_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout + weekly trend filter + volume
            if volume[i] > volume_threshold[i]:
                # Long breakout with bullish weekly trend
                if close[i] > high_max[i] and close[i] > weekly_ema50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakout with bearish weekly trend
                elif close[i] < low_min[i] and close[i] < weekly_ema50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals