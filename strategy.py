#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h primary with 1d/1w HTF - Williams %R mean reversion with weekly trend filter
    # In bear markets (2025), Williams %R > -20 signals overextended bounces to fade
    # Weekly trend filter avoids fighting the major trend (only short in weekly downtrend)
    # Target: 75-150 trades over 4 years (19-37/year) for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate Williams %R (14-period) on 1d
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R conditions for mean reversion
        # Short when overextended to upside (%R > -20) in weekly downtrend
        # Long when overextended to downside (%R < -80) in weekly uptrend
        williams_r_val = williams_r_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        close_1w_val = close_1w[min(i // (6*4*7), len(close_1w)-1)]  # approximate 1w close for trend check
        
        # Weekly trend: price above/below 50 EMA
        weekly_uptrend = close_1w_val > ema_50_val
        weekly_downtrend = close_1w_val < ema_50_val
        
        # Entry conditions
        enter_short = williams_r_val > -20 and weekly_downtrend  # fade bounce in downtrend
        enter_long = williams_r_val < -80 and weekly_uptrend     # fade crash in uptrend
        
        # Exit conditions: %R returns to mean territory (-50)
        exit_long = position == 1 and williams_r_val >= -50
        exit_short = position == -1 and williams_r_val <= -50
        
        # Execute signals
        if enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_1w_williamsr_meanrev_trend_v1"
timeframe = "6h"
leverage = 1.0