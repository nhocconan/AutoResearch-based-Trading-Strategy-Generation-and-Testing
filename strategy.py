#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # H2 = close + 0.5 * (high - low)
    # L2 = close - 0.5 * (high - low)
    # H1 = close + 0.25 * (high - low)
    # L1 = close - 0.25 * (high - low)
    
    daily_range = high_1d - low_1d
    h4 = close_1d + 1.5 * daily_range
    l4 = close_1d - 1.5 * daily_range
    h3 = close_1d + 1.0 * daily_range
    l3 = close_1d - 1.0 * daily_range
    h2 = close_1d + 0.5 * daily_range
    l2 = close_1d - 0.5 * daily_range
    h1 = close_1d + 0.25 * daily_range
    l1 = close_1d - 0.25 * daily_range
    
    # Align all levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, l2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, l1)
    
    # Volume filter: 20-period average on 4h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    # Trend filter: 50 EMA on 1d (aligned)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h2_aligned[i]) or np.isnan(l2_aligned[i]) or 
            np.isnan(h1_aligned[i]) or np.isnan(l1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long conditions: price breaks above H3 with volume and uptrend
        long_breakout = close[i] > h3_aligned[i]
        long_volume = volume_ok[i]
        long_trend = close[i] > ema_50_1d_aligned[i]
        
        # Short conditions: price breaks below L3 with volume and downtrend
        short_breakout = close[i] < l3_aligned[i]
        short_volume = volume_ok[i]
        short_trend = close[i] < ema_50_1d_aligned[i]
        
        # Exit conditions
        exit_long = close[i] < l1_aligned[i]  # Return to lower support
        exit_short = close[i] > h1_aligned[i]  # Return to upper resistance
        
        # Execute trades
        if long_breakout and long_volume and long_trend and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and short_volume and short_trend and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals