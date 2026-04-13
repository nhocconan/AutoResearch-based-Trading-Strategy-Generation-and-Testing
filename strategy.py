#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams %R mean reversion with 1w trend filter and volume confirmation.
    # Williams %R(14) < -80 = oversold long, > -20 = overbought short.
    # 1w EMA(50) filter ensures trading with weekly trend.
    # Volume spike confirms participation. Target: 50-150 total trades over 4 years.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Williams %R and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Williams %R(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r[highest_high_14 == lowest_low_14] = -50  # avoid division by zero
    
    # Calculate 1d volume MA(20) for confirmation
    volume_1d = df_1d['volume'].values
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1w EMA(50)
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume filter: current volume > 20-period MA
        volume_filter = volume[i] > volume_ma_aligned[i]
        
        # Williams %R mean reversion conditions
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # Entry conditions
        long_entry = oversold and uptrend and volume_filter
        short_entry = overbought and downtrend and volume_filter
        
        # Exit conditions: Williams %R returns to neutral zone (-50)
        long_exit = williams_r_aligned[i] > -50
        short_exit = williams_r_aligned[i] < -50
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_1d_williamsr_mean_reversion_v1"
timeframe = "12h"
leverage = 1.0