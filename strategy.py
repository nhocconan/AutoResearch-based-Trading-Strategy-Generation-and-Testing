#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Williams %R mean reversion + 1w EMA trend filter
    # Long when: Williams %R(14) < -80 (oversold) AND price > 50-period EMA(1w) (bullish trend)
    # Short when: Williams %R(14) > -20 (overbought) AND price < 50-period EMA(1w) (bearish trend)
    # Exit when: Williams %R crosses above -50 (for long) or below -50 (for short)
    # Uses discrete sizing (0.25) targeting 30-100 trades over 4 years.
    # Works in bull/bear via 1w EMA trend filter preventing counter-trend trades.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14) on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 1d Williams %R to 1d timeframe (no alignment needed as primary TF is 1d)
    williams_r_aligned = williams_r  # Already on 1d timeframe
    
    # Get 1w data for 50-period EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on 1w
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align 1w EMA to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = (williams_r_aligned[i] < -80) and (close[i] > ema_50_1w_aligned[i]) and (position != 1)
        short_entry = (williams_r_aligned[i] > -20) and (close[i] < ema_50_1w_aligned[i]) and (position != -1)
        
        # Exit conditions: Williams %R crosses -50 level
        exit_long = williams_r_aligned[i] > -50
        exit_short = williams_r_aligned[i] < -50
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
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

name = "1d_1w_williams_r_ema_trend_filter_v1"
timeframe = "1d"
leverage = 1.0