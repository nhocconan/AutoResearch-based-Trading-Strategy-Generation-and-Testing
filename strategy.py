# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Williams %R for mean-reversion entries with 1d ADX as trend filter
# - Long when price is above 1d EMA200 (bullish trend) and 12h Williams %R < -80 (oversold)
# - Short when price is below 1d EMA200 (bearish trend) and 12h Williams %R > -20 (overbought)
# - Exit when Williams %R crosses back through -50 (mean reversion complete)
# - Williams %R provides timely reversal signals; EMA200 filters for trend alignment
# - Target: 50-150 total trades over 4 years (12-38/year) for low frequency and minimal fee drag
# - Position size 0.25 for balanced risk exposure

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Load 12h data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # Calculate 12h Williams %R(14)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    wr = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    wr = np.where((highest_high - lowest_low) == 0, -50, wr)
    
    # Align indicators to 6h timeframe
    ema200_6h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    wr_6h = align_htf_to_ltf(prices, df_12h, wr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if any critical data is NaN
        if np.isnan(ema200_6h[i]) or np.isnan(wr_6h[i]):
            continue
        
        if position == 0:
            # Long: price above 1d EMA200 + Williams %R oversold (< -80)
            if (close[i] > ema200_6h[i] and 
                wr_6h[i] < -80):
                position = 1
                signals[i] = position_size
            # Short: price below 1d EMA200 + Williams %R overbought (> -20)
            elif (close[i] < ema200_6h[i] and 
                  wr_6h[i] > -20):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 (mean reversion complete)
            if wr_6h[i] > -50:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 (mean reversion complete)
            if wr_6h[i] < -50:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_1d_EMA200_12h_WilliamsR"
timeframe = "6h"
leverage = 1.0