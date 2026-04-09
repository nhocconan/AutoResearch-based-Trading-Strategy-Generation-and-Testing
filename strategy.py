#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w volume confirmation and ATR filter
# In bull markets: breakout above upper Donchian(20) captures trends
# In bear markets: breakout below lower Donchian(20) captures downtrends
# Volume confirmation and ATR filter reduce whipsaws
# Discrete position sizing 0.25 limits trades to ~10-25/year to minimize fee drag
# Works in both bull and bear markets by trading breakouts in direction of trend

name = "1d_1w_donchian_breakout_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w ATR(14) for volatility filter
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    
    # Calculate 1w average volume (20-period)
    volume_s_1w = pd.Series(volume_1w)
    avg_volume_1w = volume_s_1w.rolling(window=20, min_periods=20).mean().values
    volume_ratio_1w = np.where(atr_1w > 0, avg_volume_1w / atr_1w, np.nan)
    avg_volume_ratio_1w = pd.Series(volume_ratio_1w).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Donchian channels (20-period) based on prior close to avoid look-ahead
    # Upper = highest high of past 20 days, Lower = lowest low of past 20 days
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align 1w indicators to 1d timeframe
    avg_volume_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_volume_ratio_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(avg_volume_ratio_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price falls below midpoint of Donchian channel
            midpoint = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] < midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above midpoint of Donchian channel
            midpoint = (highest_high_20[i] + lowest_low_20[i]) / 2
            if close[i] > midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long on breakout above upper Donchian with volume confirmation
            if close[i] > highest_high_20[i] and avg_volume_ratio_1w_aligned[i] > 1.2:
                position = 1
                signals[i] = 0.25
            # Enter short on breakout below lower Donchian with volume confirmation
            elif close[i] < lowest_low_20[i] and avg_volume_ratio_1w_aligned[i] > 1.2:
                position = -1
                signals[i] = -0.25
    
    return signals