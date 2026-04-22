#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In trending markets (1d EMA50),
# we take counter-trend entries at extreme %R levels with volume spike confirmation.
# Works in both bull and bear by following 1d trend direction for entries.
# Target: 15-30 trades/year (60-120 total over 4 years) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter and Williams %R calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R (14-period) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Oversold (%R < -80) in uptrend with volume spike
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Overbought (%R > -20) in downtrend with volume spike
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral territory (-50) or opposite extreme
            if position == 1:
                if williams_r_aligned[i] >= -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if williams_r_aligned[i] <= -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_WilliamsR_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0