#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot R3/S3 breakout with volume confirmation and chop regime filter.
# Enter long when price breaks above 1d Camarilla R3 with volume spike and chop < 61.8 (trending regime).
# Enter short when price breaks below 1d Camarilla S3 with volume spike and chop < 61.8.
# Uses discrete position sizing (0.25) to balance return and drawdown. Target: 12-37 trades/year.
# Camarilla levels provide structure from higher timeframe, volume confirms breakout strength, chop filter avoids ranging markets.
# Works in bull (breakouts with trend) and bear (failed breaks reverse via exits) markets.

name = "12h_Camarilla_R3S3_Breakout_Volume_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots (HTF) - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivots (using previous bar's high, low, close)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    n_1d = len(high_1d)
    camarilla_r3 = np.full(n_1d, np.nan)
    camarilla_s3 = np.full(n_1d, np.nan)
    
    for i in range(1, n_1d):
        # Use previous bar to avoid look-ahead
        phigh = high_1d[i-1]
        plow = low_1d[i-1]
        pclose = close_1d[i-1]
        pivot = (phigh + plow + pclose) / 3.0
        rng = phigh - plow
        camarilla_r3[i] = pivot + rng * 1.1 / 4.0
        camarilla_s3[i] = pivot - rng * 1.1 / 4.0
    
    # Forward fill Camarilla levels
    camarilla_r3 = pd.Series(camarilla_r3).ffill().values
    camarilla_s3 = pd.Series(camarilla_s3).ffill().values
    
    # Align 1d indicators to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 12h chop regime: EHLERS CHOPPINESS INDEX (14)
    def choppiness_index(high, low, close, length=14):
        atr_sum = np.zeros_like(close)
        true_range = np.zeros_like(close)
        for i in range(1, len(close)):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            true_range[i] = tr
            if i >= length:
                atr_sum[i] = atr_sum[i-1] + tr - true_range[i-length+1]
            else:
                atr_sum[i] = atr_sum[i-1] + tr
        atr = atr_sum / length
        max_high = np.zeros_like(close)
        min_low = np.zeros_like(close)
        for i in range(len(close)):
            if i < length:
                max_high[i] = np.max(high[:i+1])
                min_low[i] = np.min(low[:i+1])
            else:
                max_high[i] = np.max(high[i-length+1:i+1])
                min_low[i] = np.min(low[i-length+1:i+1])
        chop = np.zeros_like(close)
        for i in range(length-1, len(close)):
            if max_high[i] != min_low[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(length)
            else:
                chop[i] = 50.0
        return chop
    
    chop = choppiness_index(high, low, close, 14)
    chop_trending = chop < 61.8  # Trending regime when chop < 61.8
    
    # Calculate 12h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions with volume confirmation and chop filter
        long_breakout = close[i] > camarilla_r3_aligned[i] and volume_spike[i] and chop_trending[i]
        short_breakout = close[i] < camarilla_s3_aligned[i] and volume_spike[i] and chop_trending[i]
        
        # Exit conditions: opposite Camarilla level
        long_exit = close[i] < camarilla_s3_aligned[i]
        short_exit = close[i] > camarilla_r3_aligned[i]
        
        # Handle entries and exits
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals