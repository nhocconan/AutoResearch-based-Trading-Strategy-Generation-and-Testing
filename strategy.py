#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d Williams %R reversal and 1w EMA trend filter
# Williams %R identifies overbought/oversold conditions on daily timeframe
# EMA on weekly timeframe determines primary trend direction
# Combines mean reversion in ranging markets with trend following in trending markets
# Works in both bull and bear markets by adapting to regime

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for Williams %R
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams %R (14 periods)
    wr_length = 14
    highest_high = pd.Series(df_1d['high']).rolling(window=wr_length, min_periods=wr_length).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=wr_length, min_periods=wr_length).min().values
    # Avoid division by zero
    diff = highest_high - lowest_low
    wr = np.where(diff != 0, -100 * (highest_high - df_1d['close'].values) / diff, -50)
    
    # Align Williams %R to 12h timeframe
    wr_aligned = align_htf_to_ltf(prices, df_1d, wr)
    
    # Load 1w data ONCE for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA (21 periods) on weekly close
    ema_length = 21
    ema = pd.Series(df_1w['close']).ewm(span=ema_length, adjust=False, min_periods=ema_length).mean().values
    
    # Align EMA to 12h timeframe
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, wr_length, ema_length)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(wr_aligned[i]) or 
            np.isnan(ema_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Williams %R levels
        wr_oversold = -80  # Oversold threshold
        wr_overbought = -20  # Overbought threshold
        
        if position == 0:
            # Enter long: Williams %R oversold AND price above weekly EMA (uptrend)
            if wr_aligned[i] <= wr_oversold and price > ema_aligned[i]:
                position = 1
                signals[i] = position_size
            # Enter short: Williams %R overbought AND price below weekly EMA (downtrend)
            elif wr_aligned[i] >= wr_overbought and price < ema_aligned[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R becomes overbought OR price crosses below weekly EMA
            if wr_aligned[i] >= wr_overbought or price < ema_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R becomes oversold OR price crosses above weekly EMA
            if wr_aligned[i] <= wr_oversold or price > ema_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dWR_1wEMA_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0