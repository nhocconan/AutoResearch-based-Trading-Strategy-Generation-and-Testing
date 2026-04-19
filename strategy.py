#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1w/1d confluence.
# Uses 1w EMA50 for primary trend and 1d Williams %R for mean reversion entries.
# Enters only when price pulls back to extreme levels in trending context.
# Targets 15-30 trades/year (60-120 total over 4 years) with strict entry conditions.
# Works in bull/bear by following higher timeframe trend direction.
name = "6h_1w_EMA50_1d_WilliamsR_MeanReversion"
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
    
    # Pre-compute session filter (08-20 UTC) - trade during active hours
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for EMA50 trend (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Williams %R (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume filter: volume > 1.2 * 20-period average (less strict for 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_ma[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: uptrend (price > weekly EMA50) + oversold (Williams %R < -80) + volume
            if (close[i] > ema_50_1w_aligned[i] and 
                williams_r_aligned[i] < -80 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: downtrend (price < weekly EMA50) + overbought (Williams %R > -20) + volume
            elif (close[i] < ema_50_1w_aligned[i] and 
                  williams_r_aligned[i] > -20 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if trend changes or overbought condition reached
            if (close[i] < ema_50_1w_aligned[i] or 
                williams_r_aligned[i] > -20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if trend changes or oversold condition reached
            if (close[i] > ema_50_1w_aligned[i] or 
                williams_r_aligned[i] < -80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals