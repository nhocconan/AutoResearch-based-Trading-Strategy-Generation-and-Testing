#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Supertrend for trend direction and 1h Donchian breakout for entry timing.
# Enter long when 4h Supertrend is bullish and price breaks above 1h Donchian upper band (20).
# Enter short when 4h Supertrend is bearish and price breaks below 1h Donchian lower band (20).
# Uses discrete position sizing (0.20) to minimize fee churn. Target: 15-35 trades/year.
# Supertrend provides reliable trend filtering from higher timeframe, Donchian breakouts capture momentum.
# Works in bull (trend continuation breaks) and bear (trend reversal breaks) markets.

name = "1h_Supertrend4h_DonchianBreakout_20_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Supertrend (HTF)
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Supertrend (10, 3.0)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr = np.zeros(len(close_4h))
    for i in range(1, len(close_4h)):
        tr[i] = max(high_4h[i] - low_4h[i], 
                    abs(high_4h[i] - close_4h[i-1]), 
                    abs(low_4h[i] - close_4h[i-1]))
    tr[0] = high_4h[0] - low_4h[0]
    
    # ATR(10)
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_4h + low_4h) / 2 + 3.0 * atr
    basic_lb = (high_4h + low_4h) / 2 - 3.0 * atr
    
    # Final Upper and Lower Bands
    final_ub = np.zeros(len(close_4h))
    final_lb = np.zeros(len(close_4h))
    for i in range(len(close_4h)):
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            if basic_ub[i] < final_ub[i-1] or close_4h[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
                
            if basic_lb[i] > final_lb[i-1] or close_4h[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
    
    # Supertrend
    supertrend = np.zeros(len(close_4h))
    for i in range(len(close_4h)):
        if i == 0:
            supertrend[i] = final_ub[i]
        else:
            if supertrend[i-1] == final_ub[i-1] and close_4h[i] <= final_ub[i]:
                supertrend[i] = final_ub[i]
            elif supertrend[i-1] == final_ub[i-1] and close_4h[i] > final_ub[i]:
                supertrend[i] = final_lb[i]
            elif supertrend[i-1] == final_lb[i-1] and close_4h[i] >= final_lb[i]:
                supertrend[i] = final_lb[i]
            elif supertrend[i-1] == final_lb[i-1] and close_4h[i] < final_lb[i]:
                supertrend[i] = final_ub[i]
    
    # Supertrend direction: 1 = bullish (price above supertrend), -1 = bearish (price below)
    supertrend_dir = np.where(close_4h > supertrend, 1, -1)
    
    # Align 4h Supertrend direction to 1h timeframe
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_4h, supertrend_dir.astype(float))
    
    # Calculate 1h Donchian Channel (20)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = donchian_period  # Ensure sufficient history for Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_dir_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions with 4h Supertrend filter
        long_breakout = (supertrend_dir_aligned[i] == 1 and 
                         close[i] > highest_high[i])
        short_breakout = (supertrend_dir_aligned[i] == -1 and 
                          close[i] < lowest_low[i])
        
        # Exit conditions: Donchian opposite band
        long_exit = close[i] < lowest_low[i]
        short_exit = close[i] > highest_high[i]
        
        # Handle entries and exits
        if long_breakout and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.20
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals