#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day True Range to detect volatility breakouts in the direction of the 1-day EMA trend.
# Enters long when price breaks above prior day's high + 0.5*ATR(14) with 1-day uptrend, short when price breaks below prior day's low - 0.5*ATR(14) with 1-day downtrend.
# Uses volatility expansion as entry signal, which works in both trending and mean-reverting markets by capturing momentum bursts.
# Target: 20-40 trades/year to minimize fee drag while capturing significant moves.

name = "4h_VolatilityBreakout_1dTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volatility calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA20 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate 1-day ATR(14) for volatility measurement
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = np.roll(close_1d, 1)
    close_1d_prev[0] = close_1d[0]  # first value
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d_prev)
    tr3 = np.abs(low_1d - close_1d_prev)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Prior day's high and low for breakout levels
    prior_high_1d = np.roll(high_1d, 1)
    prior_low_1d = np.roll(low_1d, 1)
    # Handle first value
    prior_high_1d[0] = high_1d[0]
    prior_low_1d[0] = low_1d[0]
    
    prior_high_aligned = align_htf_to_ltf(prices, df_1d, prior_high_1d)
    prior_low_aligned = align_htf_to_ltf(prices, df_1d, prior_low_1d)
    
    # Breakout levels: prior day's high/low +/- 0.5 * ATR(14)
    breakout_up = prior_high_aligned + 0.5 * atr14_1d_aligned
    breakout_down = prior_low_aligned - 0.5 * atr14_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Need enough data for EMA20 and ATR14
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema20_1d_aligned[i]) or 
            np.isnan(atr14_1d_aligned[i]) or
            np.isnan(breakout_up[i]) or
            np.isnan(breakout_down[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema20_1d_val = ema20_1d_aligned[i]
        up_break = breakout_up[i]
        down_break = breakout_down[i]
        
        if position == 0:
            # Enter long: Price breaks above prior day's high + 0.5*ATR with 1-day uptrend
            if close[i] > up_break and close[i] > ema20_1d_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below prior day's low - 0.5*ATR with 1-day downtrend
            elif close[i] < down_break and close[i] < ema20_1d_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below prior day's low or trend turns down
            if close[i] < prior_low_aligned[i] or close[i] < ema20_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above prior day's high or trend turns up
            if close[i] > prior_high_aligned[i] or close[i] > ema20_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals