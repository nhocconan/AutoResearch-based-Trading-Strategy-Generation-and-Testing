#!/usr/bin/env python3
"""
6h_1w1d_MarketRegime_Filter_TrendFollow
Hypothesis: On 6-hour timeframe, follow trend only when weekly and daily regimes align.
Weekly regime: price above/below weekly 200 EMA (bull/bear).
Daily regime: price above/below daily 50 EMA (bull/bear).
Entry: price breaks above/below 6h Donchian(20) channel in direction of aligned weekly/daily trend.
Exit: price returns to 6h EMA(50) or opposite Donchian break.
Designed for low trade frequency (~15-25/year) by requiring multi-timeframe alignment.
Works in bull (trend follow) and bear (avoids counter-trend trades via regime filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for regime filter and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly 200 EMA for regime
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Daily 50 EMA for regime
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h EMA(50) for exit
    ema_50_6h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_50_6h[i])):
            signals[i] = 0.0
            continue
        
        # Determine regimes
        weekly_bull = close[i] > ema_200_1w_aligned[i]
        weekly_bear = close[i] < ema_200_1w_aligned[i]
        daily_bull = close[i] > ema_50_1d_aligned[i]
        daily_bear = close[i] < ema_50_1d_aligned[i]
        
        # Aligned regimes (both agree)
        bull_regime = weekly_bull and daily_bull
        bear_regime = weekly_bear and daily_bear
        
        # Donchian breakout conditions
        donchian_break_up = high[i] > high_20[i-1]  # Break above previous period's high
        donchian_break_down = low[i] < low_20[i-1]  # Break below previous period's low
        
        # Exit conditions
        long_exit = (position == 1) and (close[i] < ema_50_6h[i] or donchian_break_down)
        short_exit = (position == -1) and (close[i] > ema_50_6h[i] or donchian_break_up)
        
        # Entry logic
        if bull_regime and donchian_break_up and position <= 0:
            signals[i] = 0.25
            position = 1
        elif bear_regime and donchian_break_down and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "6h_1w1d_MarketRegime_Filter_TrendFollow"
timeframe = "6h"
leverage = 1.0