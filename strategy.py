#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extreme readings with volume confirmation and chop regime filter.
# Enter long when 1d Williams %R < -80 (oversold) with volume spike and chop < 61.8 (trending regime).
# Enter short when 1d Williams %R > -20 (overbought) with volume spike and chop < 61.8.
# Uses discrete position sizing (0.25) to balance return and drawdown. Target: 20-50 trades/year.
# Williams %R identifies overextended moves, volume confirms breakout strength, chop filter avoids ranging markets.
# Works in bull (mean reversion from oversold) and bear (failed rallies from overbought) markets.

name = "4h_WilliamsR_1d_Extreme_Volume_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    n_1d = len(high_1d)
    williams_r = np.full(n_1d, np.nan)
    
    for i in range(13, n_1d):  # min_periods=14
        highest_high = np.max(high_1d[i-13:i+1])
        lowest_low = np.min(low_1d[i-13:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close_1d[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50.0
    
    # Forward fill Williams %R
    williams_r = pd.Series(williams_r).ffill().values
    
    # Align 1d indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 4h chop regime: EHLERS CHOPPINESS INDEX (14)
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
    
    # Calculate 4h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R extreme conditions with volume confirmation and chop filter
        long_signal = williams_r_aligned[i] < -80 and volume_spike[i] and chop_trending[i]
        short_signal = williams_r_aligned[i] > -20 and volume_spike[i] and chop_trending[i]
        
        # Exit conditions: opposite extreme
        long_exit = williams_r_aligned[i] > -20
        short_exit = williams_r_aligned[i] < -80
        
        # Handle entries and exits
        if long_signal and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_signal and position >= 0:
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