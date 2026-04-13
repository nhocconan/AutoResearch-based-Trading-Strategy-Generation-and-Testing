#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h primary with 1w HTF - Donchian breakout with 1w trend filter
    # Designed to capture medium-term trend continuations with low trade frequency
    # Works in bull markets via breakouts above Donchian upper band
    # Works in bear markets via breakouts below Donchian lower band with 1w trend filter
    # Target: 12-37 trades/year (50-150 total) for minimal fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for HTF Donchian channels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channels (20-period)
    donchian_h_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_l_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF indicators to 12h primary timeframe
    donchian_h_aligned = align_htf_to_ltf(prices, df_1w, donchian_h_20)
    donchian_l_aligned = align_htf_to_ltf(prices, df_1w, donchian_l_20)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_h_aligned[i]) or 
            np.isnan(donchian_l_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions at 1w Donchian levels
        breakout_up = high[i] > donchian_h_aligned[i]
        breakout_down = low[i] < donchian_l_aligned[i]
        
        # Trend filter: only trade in direction of 1w EMA50
        trend_filter_long = close[i] > ema50_1w_aligned[i]
        trend_filter_short = close[i] < ema50_1w_aligned[i]
        
        # Entry conditions
        enter_long = breakout_up and trend_filter_long
        enter_short = breakout_down and trend_filter_short
        
        # Exit conditions: opposite Donchian breakout
        exit_long = position == 1 and low[i] < donchian_l_aligned[i]
        exit_short = position == -1 and high[i] > donchian_h_aligned[i]
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
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

name = "12h_1w_donchian_breakout_trend_v1"
timeframe = "12h"
leverage = 1.0