#!/usr/bin/env python3
"""
1d_Weekly_Donchian_Breakout_Trend_Filter
Hypothesis: Weekly Donchian breakouts (20-week) on 1d timeframe, filtered by weekly EMA(34) trend, capture major trends in both bull and bear markets. Weekly timeframe reduces noise, trend filter avoids counter-trend trades, and breakouts capture momentum. Target: 10-25 trades per year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for Donchian channels and EMA (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high']
    low_1w = df_1w['low']
    
    # Upper band: highest high over past 20 weeks
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over past 20 weeks
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use previous week's levels only (avoid look-ahead)
    upper_20_prev = upper_20[:-1]  # remove last element
    lower_20_prev = lower_20[:-1]  # remove last element
    # Prepend NaN for first week
    upper_20_prev = np.concatenate([[np.nan], upper_20_prev])
    lower_20_prev = np.concatenate([[np.nan], lower_20_prev])
    
    # Align to daily timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20_prev)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20_prev)
    
    # Get weekly EMA(34) for trend filter
    close_1w = df_1w['close']
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # sufficient warmup for weekly indicators
    
    for i in range(start_idx, n):
        if (np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_20_aligned[i]
        lower = lower_20_aligned[i]
        ema_trend = ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: break above weekly upper band with price above weekly EMA (uptrend)
            if price > upper and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly lower band with price below weekly EMA (downtrend)
            elif price < lower and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long: exit when price breaks below weekly lower band or crosses below weekly EMA
            if price < lower or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short: exit when price breaks above weekly upper band or crosses above weekly EMA
            if price > upper or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Donchian_Breakout_Trend_Filter"
timeframe = "1d"
leverage = 1.0