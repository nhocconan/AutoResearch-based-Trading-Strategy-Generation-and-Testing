#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1d Williams %R for mean reversion in ranging markets,
# 1d EMA200 for trend filter, and volume confirmation. Enters only during 08-20 UTC session.
# Uses contrarian entries when price is oversold/overbought but aligned with higher timeframe trend.
# Targets 20-40 trades/year (80-160 total over 4 years) with strict entry conditions.
# Works in bull/bear by combining trend filter with mean reversion entries.
name = "6h_1d_WilliamsR_EMA200_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Williams %R and EMA200 (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)) * -100
    
    # EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(volume_ma[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: oversold (Williams %R < -80) AND price above EMA200 (uptrend) with volume
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_200_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: overbought (Williams %R > -20) AND price below EMA200 (downtrend) with volume
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_200_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Williams %R returns to neutral territory (> -50) or price breaks below EMA200
            if williams_r_aligned[i] > -50 or close[i] < ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Williams %R returns to neutral territory (< -50) or price breaks above EMA200
            if williams_r_aligned[i] < -50 or close[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals