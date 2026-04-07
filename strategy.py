#!/usr/bin/env python3
"""
1d_weekly_pivot_reversion_v2
Hypothesis: Weekly pivot levels act as strong support/resistance on 1d timeframe. Price tends to revert from R1/S1 towards pivot point (PP). In ranging markets (2025-2026), this mean reversion works well. Uses weekly PP, R1, S1 as reference. Enter long when price crosses above S1 with bullish momentum (close > open) and weekly trend up (price > weekly MA50). Enter short when price crosses below R1 with bearish momentum (close < open) and weekly trend down (price < weekly MA50). Weekly trend filter prevents counter-trend trades. Targets 15-25 trades/year to minimize fee drag. Works in bull via trend filter and in bear via mean reversion from overextended levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_pivot_reversion_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    
    # Calculate weekly pivot points (PP, R1, S1) from weekly OHLC
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Weekly high, low, close
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Pivot point calculation: PP = (H + L + C) / 3
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    # R1 = 2 * PP - L
    r1 = 2 * pp - weekly_low
    # S1 = 2 * PP - H
    s1 = 2 * pp - weekly_high
    
    # Align weekly levels to daily (with shift(1) for completed weekly bar only)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # Weekly trend filter: 50-period MA on weekly close
    weekly_ma50 = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    weekly_ma50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ma50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Need 50 periods for weekly MA50
        # Skip if required data not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(weekly_ma50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Bullish/bearish momentum from daily candle
        bullish_momentum = close[i] > open_price[i]
        bearish_momentum = close[i] < open_price[i]
        
        # Weekly trend direction
        weekly_uptrend = close[i] > weekly_ma50_aligned[i]
        weekly_downtrend = close[i] < weekly_ma50_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price crosses below pivot point (mean reversion complete)
            if close[i] < pp_aligned[i]:
                exit_long = True
            # Exit if weekly trend turns down
            elif weekly_downtrend:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price crosses above pivot point (mean reversion complete)
            if close[i] > pp_aligned[i]:
                exit_short = True
            # Exit if weekly trend turns up
            elif weekly_uptrend:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price crosses above S1 with bullish momentum and weekly uptrend
            long_entry = False
            if (close[i] > s1_aligned[i] and close[i-1] <= s1_aligned[i-1] and
                bullish_momentum and weekly_uptrend):
                long_entry = True
            
            # Short entry: price crosses below R1 with bearish momentum and weekly downtrend
            short_entry = False
            if (close[i] < r1_aligned[i] and close[i-1] >= r1_aligned[i-1] and
                bearish_momentum and weekly_downtrend):
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals