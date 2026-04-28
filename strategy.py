#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Supertrend for trend direction and 1d ATR-based volatility breakout for entry.
# Enter long when price breaks above 1d high + 0.5*ATR(20) with 12h Supertrend uptrend and volume confirmation.
# Enter short when price breaks below 1d low - 0.5*ATR(20) with 12h Supertrend downtrend and volume confirmation.
# Uses discrete position sizing (0.25) and ATR-based stoploss. Target: 20-50 trades/year.
# Supertrend provides reliable trend filter, ATR breakout captures volatility expansion, volume confirms strength.
# Works in bull (trend continuation breaks) and bear (trend reversal breaks) markets.

name = "4h_Supertrend12h_ATRBreakout_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Supertrend (HTF)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Supertrend
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    n_12h = len(high_12h)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr = np.zeros(n_12h)
    for i in range(1, n_12h):
        tr[i] = max(high_12h[i] - low_12h[i], 
                   abs(high_12h[i] - close_12h[i-1]), 
                   abs(low_12h[i] - close_12h[i-1]))
    tr[0] = high_12h[0] - low_12h[0]
    
    # ATR
    atr = np.zeros(n_12h)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, n_12h):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Supertrend calculation
    supertrend = np.zeros(n_12h)
    direction = np.ones(n_12h)  # 1 for uptrend, -1 for downtrend
    
    # Basic upper and lower bands
    basic_ub = (high_12h + low_12h) / 2 + multiplier * atr
    basic_lb = (high_12h + low_12h) / 2 - multiplier * atr
    
    # Final upper and lower bands
    final_ub = np.zeros(n_12h)
    final_lb = np.zeros(n_12h)
    
    for i in range(atr_period, n_12h):
        if close_12h[i-1] <= final_ub[i-1]:
            final_ub[i] = min(basic_ub[i], final_ub[i-1])
        else:
            final_ub[i] = basic_ub[i]
            
        if close_12h[i-1] >= final_lb[i-1]:
            final_lb[i] = max(basic_lb[i], final_lb[i-1])
        else:
            final_lb[i] = basic_lb[i]
    
    # Supertrend and direction
    for i in range(atr_period, n_12h):
        if close_12h[i] <= final_ub[i]:
            supertrend[i] = final_ub[i]
            direction[i] = -1
        else:
            supertrend[i] = final_lb[i]
            direction[i] = 1
    
    # Forward fill for initialization period
    for i in range(1, atr_period):
        direction[i] = direction[i-1]
        supertrend[i] = supertrend[i-1] if i < len(supertrend) and not np.isnan(supertrend[i-1]) else 0
        final_ub[i] = final_ub[i-1] if i < len(final_ub) and not np.isnan(final_ub[i-1]) else basic_ub[i]
        final_lb[i] = final_lb[i-1] if i < len(final_lb) and not np.isnan(final_lb[i-1]) else basic_lb[i]
    
    # Align 12h Supertrend direction to 4h timeframe
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # Get 1d data for ATR-based breakout levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR for breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    n_1d = len(high_1d)
    atr_period_1d = 20
    
    # True Range for 1d
    tr_1d = np.zeros(n_1d)
    for i in range(1, n_1d):
        tr_1d[i] = max(high_1d[i] - low_1d[i], 
                      abs(high_1d[i] - close_1d[i-1]), 
                      abs(low_1d[i] - close_1d[i-1]))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # ATR for 1d
    atr_1d = np.zeros(n_1d)
    atr_1d[atr_period_1d-1] = np.mean(tr_1d[:atr_period_1d])
    for i in range(atr_period_1d, n_1d):
        atr_1d[i] = (atr_1d[i-1] * (atr_period_1d-1) + tr_1d[i]) / atr_period_1d
    
    # Forward fill ATR for initialization period
    for i in range(1, atr_period_1d):
        atr_1d[i] = atr_1d[i-1]
    
    # Calculate breakout levels: previous 1d high/low ± 0.5*ATR
    breakout_up = np.full(n_1d, np.nan)
    breakout_down = np.full(n_1d, np.nan)
    
    for i in range(1, n_1d):
        breakout_up[i] = high_1d[i-1] + 0.5 * atr_1d[i-1]
        breakout_down[i] = low_1d[i-1] - 0.5 * atr_1d[i-1]
    
    # Forward fill breakout levels
    breakout_up = pd.Series(breakout_up).ffill().values
    breakout_down = pd.Series(breakout_down).ffill().values
    
    # Align 1d breakout levels to 4h timeframe
    breakout_up_aligned = align_htf_to_ltf(prices, df_1d, breakout_up)
    breakout_down_aligned = align_htf_to_ltf(prices, df_1d, breakout_down)
    
    # Calculate 4h volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(direction_aligned[i]) or np.isnan(breakout_up_aligned[i]) or 
            np.isnan(breakout_down_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions with trend filter and volume confirmation
        long_breakout = close[i] > breakout_up_aligned[i] and direction_aligned[i] == 1 and volume_confirm[i]
        short_breakout = close[i] < breakout_down_aligned[i] and direction_aligned[i] == -1 and volume_confirm[i]
        
        # Exit conditions: opposite breakout level or trend change
        long_exit = close[i] < breakout_down_aligned[i] or direction_aligned[i] == -1
        short_exit = close[i] > breakout_up_aligned[i] or direction_aligned[i] == 1
        
        # Handle entries and exits
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals