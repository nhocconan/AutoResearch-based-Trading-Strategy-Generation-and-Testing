#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy combining 1d Williams %R for mean reversion signals and 1w ADX for trend strength
# Williams %R < -80 indicates oversold, > -20 indicates overbought
# ADX > 25 confirms trending conditions to avoid counter-trend trades
# Williams %R cross above -50 from below signals long entry in uptrend
# Williams %R cross below -50 from above signals short entry in downtrend
# Uses weekly ADX for regime filter and daily Williams %R for entry timing - balances signal frequency
# Designed for 12h timeframe to target 12-37 trades per year (50-150 total over 4 years)

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
    wr_len = 14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Highest High and Lowest Low over lookback period
    highest_high = pd.Series(high_1d).rolling(window=wr_len, min_periods=wr_len).max().values
    lowest_low = pd.Series(low_1d).rolling(window=wr_len, min_periods=wr_len).min().values
    
    # Williams %R formula: (Highest High - Close) / (Highest High - Lowest Low) * -100
    wr = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    wr = np.where((highest_high - lowest_low) == 0, -50, wr)  # Avoid division by zero
    
    # Align Williams %R to 12h timeframe
    wr_aligned = align_htf_to_ltf(prices, df_1d, wr)
    
    # Load 1w data ONCE for ADX
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w ADX (14 periods)
    adx_len = 14
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=adx_len, min_periods=adx_len).sum().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_sum / tr_sum
    minus_di = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=adx_len, min_periods=adx_len).mean().values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, wr_len + adx_len)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(wr_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Williams %R signals
        oversold = wr_aligned[i] < -80
        overbought = wr_aligned[i] > -20
        
        if position == 0:
            # Enter long: Williams %R crosses above -50 from below in trending market
            if i > start and wr_aligned[i-1] < -50 <= wr_aligned[i] and trending:
                position = 1
                signals[i] = position_size
            # Enter short: Williams %R crosses below -50 from above in trending market
            elif i > start and wr_aligned[i-1] > -50 >= wr_aligned[i] and trending:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) or ADX weakens
            if wr_aligned[i] > -20 or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) or ADX weakens
            if wr_aligned[i] < -80 or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dWR_1wADX_Trend_Momentum_v1"
timeframe = "12h"
leverage = 1.0