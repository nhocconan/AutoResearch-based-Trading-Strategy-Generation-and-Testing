#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 12h EMA34 trend and 1d Williams %R mean reversion.
# Uses 12h EMA34 for trend direction and 1d Williams %R for oversold/overbought signals.
# Enters only during 08-20 UTC session to avoid low-volume noise.
# Targets 15-37 trades/year (60-150 total over 4 years) with strict entry conditions.
# Works in bull/bear by following higher timeframe trends and fading extremes.
name = "6h_12hEMA34_1dWilliamsR_MeanReversion"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for EMA34 trend (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Get 1d data for Williams %R (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    # Handle division by zero
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(williams_r_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 12h EMA34 AND Williams %R oversold (< -80)
            if (close[i] > ema_34_12h_aligned[i] and 
                williams_r_aligned[i] < -80):
                signals[i] = 0.25
                position = 1
            # Short: price below 12h EMA34 AND Williams %R overbought (> -20)
            elif (close[i] < ema_34_12h_aligned[i] and 
                  williams_r_aligned[i] > -20):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below 12h EMA34 or Williams %R overbought (> -20)
            if close[i] < ema_34_12h_aligned[i] or williams_r_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above 12h EMA34 or Williams %R oversold (< -80)
            if close[i] > ema_34_12h_aligned[i] or williams_r_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals