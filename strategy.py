#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R mean reversion with 1d trend filter
    # Long: Williams %R < -80 (oversold) + price > 1d EMA50 (uptrend)
    # Short: Williams %R > -20 (overbought) + price < 1d EMA50 (downtrend)
    # Exit: Williams %R crosses -50 (mean reversion midpoint)
    # Uses 1d EMA for trend alignment to avoid counter-trend trades
    # Williams %R identifies extreme price exhaustion points
    # Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend)
    # Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and fees
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 6h data for primary timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get 1d data for EMA trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 6h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_6h) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(14, n):  # start from 14 to have enough data for Williams %R
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        exit_signal = williams_r[i] > -50  # exit when crosses above -50
        
        # Trend filter from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        long_entry = oversold and uptrend and position != 1
        short_entry = overbought and downtrend and position != -1
        
        # Exit conditions
        exit_long = position == 1 and exit_signal
        exit_short = position == -1 and exit_signal
        
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

name = "6h_1d_williamsr_trend_filter_v1"
timeframe = "6h"
leverage = 1.0