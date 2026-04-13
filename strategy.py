#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1-day Williams %R mean reversion + 1-week trend filter
# Long when 1d Williams %R < -80 (oversold) and price > 1w EMA200 (bullish long-term trend)
# Short when 1d Williams %R > -20 (overbought) and price < 1w EMA200 (bearish long-term trend)
# Exit when Williams %R crosses -50 (mean reversion complete)
# Williams %R identifies exhaustion points; EMA200 filter ensures alignment with major trend
# Target: 60-120 total trades over 4 years (15-30/year) to balance opportunity and cost
# Uses weekly EMA to avoid counter-trend trades in strong trends, improving win rate

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1-day data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Get 1-week data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align 1-day Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Align 1-week EMA200 to 6h timeframe
    ema200_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema200_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        price = close[i]
        ema200 = ema200_aligned[i]
        
        # Entry conditions
        long_entry = (wr < -80) and (price > ema200)
        short_entry = (wr > -20) and (price < ema200)
        
        # Exit conditions: Williams %R crosses -50 (mean reversion midpoint)
        exit_long = (position == 1) and (wr > -50)
        exit_short = (position == -1) and (wr < -50)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
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

name = "6h_1d_1w_williams_r_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0