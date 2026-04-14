#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1d Williams %R overbought/oversold and 1w EMA trend filter
# Williams %R identifies overbought (> -20) and oversold (< -80) conditions
# When price is oversold in an uptrend (price > 1w EMA) or overbought in a downtrend (price < 1w EMA),
# we take mean-reversion trades with the trend
# Uses 1d Williams %R for entry and 1w EMA for trend filter - avoids overtrading by requiring trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for Williams %R
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Williams %R (14 periods)
    wr_length = 14
    highest_high = pd.Series(df_1d['high']).rolling(window=wr_length, min_periods=wr_length).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=wr_length, min_periods=wr_length).min().values
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    wr = (highest_high - df_1d['close'].values) / (highest_high - lowest_low) * -100
    wr = np.where((highest_high - lowest_low) == 0, -50, wr)  # Handle division by zero
    
    # Align Williams %R to 1d timeframe (no alignment needed as we're already on 1d)
    wr_aligned = wr  # Already on 1d timeframe
    
    # Load 1w data ONCE for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA (50 periods)
    ema_length = 50
    ema_1w = pd.Series(df_1w['close']).ewm(span=ema_length, adjust=False, min_periods=ema_length).mean().values
    
    # Align 1w EMA to 1d timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 50)  # Need enough for Williams %R and EMA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(wr_aligned[i]) or 
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Williams %R levels
        wr_value = wr_aligned[i]
        oversold = wr_value < -80
        overbought = wr_value > -20
        
        # Trend filter: price vs 1w EMA
        uptrend = price > ema_1w_aligned[i]
        downtrend = price < ema_1w_aligned[i]
        
        if position == 0:
            # Enter long: oversold in uptrend OR overbought in strong uptrend (momentum)
            if oversold and uptrend:
                position = 1
                signals[i] = position_size
            # Enter short: overbought in downtrend OR oversold in strong downtrend (momentum)
            elif overbought and downtrend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: overbought OR trend breakdown
            if overbought or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: oversold OR trend reversal
            if oversold or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1dWR_1wEMA_TrendMeanRev_v1"
timeframe = "1d"
leverage = 1.0