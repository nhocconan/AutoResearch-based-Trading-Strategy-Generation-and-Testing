#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Williams %R for mean reversion + volume filter
# Long when daily Williams %R crosses above -80 (oversold) with volume > 1.2x 20-period avg
# Short when daily Williams %R crosses below -20 (overbought) with volume > 1.2x 20-period avg
# Williams %R measures momentum and identifies overbought/oversold conditions
# Works in both bull and bear markets by fading extremes with volume confirmation
# Target: 20-40 trades per year (80-160 over 4 years) with 0.25 position sizing

name = "4h_1dWilliamsR_MeanReversion_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    
    # Align Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume filter: >1.2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.2 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Williams %R warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r_aligned[i]
        wr_prev = williams_r_aligned[i-1]
        
        if position == 0:
            # Long entry: Williams %R crosses above -80 from below (bullish reversal from oversold)
            if wr > -80 and wr_prev <= -80 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -20 from above (bearish reversal from overbought)
            elif wr < -20 and wr_prev >= -20 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -50 (momentum fading)
            if wr < -50 and wr_prev >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -50 (momentum fading)
            if wr > -50 and wr_prev <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals