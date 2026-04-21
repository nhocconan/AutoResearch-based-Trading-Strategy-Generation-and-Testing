#!/usr/bin/env python3
"""
6h_12h_1d_1w_Camarilla_R1S1_Breakout_Pullback_V1
Hypothesis: Buy pullbacks to EMA34 after 12h breakouts of daily R1/S1 with weekly trend filter.
Works in bull/bear by trading with weekly trend. Pullbacks improve risk/reward vs breakout chase.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    # Load 12h data for breakout detection
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Load 1d data for daily Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (R1, S1, PP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    rang = prev_high - prev_low
    r1 = prev_close + 1.1 * rang / 12
    s1 = prev_close - 1.1 * rang / 12
    pp = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to 12h timeframe
    r1_12h = align_htf_to_ltf(df_12h['close'], df_1d, r1)
    s1_12h = align_htf_to_ltf(df_12h['close'], df_1d, s1)
    pp_12h = align_htf_to_ltf(df_12h['close'], df_1d, pp)
    
    # Calculate 12h breakout signals (close above R1 or below S1)
    close_12h = df_12h['close'].values
    breakout_up = close_12h > r1_12h
    breakout_down = close_12h < s1_12h
    
    # Align breakout signals to 6h timeframe
    breakout_up_6h = align_htf_to_ltf(prices, df_12h, breakout_up)
    breakout_down_6h = align_htf_to_ltf(prices, df_12h, breakout_down)
    
    # Calculate EMA34 on 6h for pullback entries
    close_6h = prices['close'].values
    ema34_6h = pd.Series(close_6h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_6h = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    breakout_active_up = False
    breakout_active_down = False
    
    for i in range(80, n):
        # Skip if indicators not ready
        if (np.isnan(breakout_up_6h[i]) or np.isnan(breakout_down_6h[i]) or
            np.isnan(ema34_6h[i]) or np.isnan(ema34_1w_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                breakout_active_up = False
                breakout_active_down = False
            continue
        
        price = close_6h[i]
        ema34 = ema34_6h[i]
        weekly_ema = ema34_1w_6h[i]
        
        # Update breakout flags (persist until opposite breakout)
        if breakout_up_6h[i]:
            breakout_active_up = True
            breakout_active_down = False
        if breakout_down_6h[i]:
            breakout_active_down = True
            breakout_active_up = False
        
        # Weekly trend filter
        weekly_uptrend = weekly_ema > ema34_1w_6h[i-1] if i > 80 else False
        weekly_downtrend = weekly_ema < ema34_1w_6h[i-1] if i > 80 else False
        
        if position == 0:
            # Long: pullback to EMA34 after upward breakout in weekly uptrend
            if breakout_active_up and weekly_uptrend and price <= ema34 * 1.005:
                signals[i] = 0.25
                position = 1
                breakout_active_up = False  # reset after entry
            # Short: pullback to EMA34 after downward breakout in weekly downtrend
            elif breakout_active_down and weekly_downtrend and price >= ema34 * 0.995:
                signals[i] = -0.25
                position = -1
                breakout_active_down = False  # reset after entry
        
        elif position == 1:
            # Long exit: price breaks below EMA34 or weekly trend changes
            if price < ema34 * 0.995 or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
                breakout_active_up = False
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above EMA34 or weekly trend changes
            if price > ema34 * 1.005 or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
                breakout_active_down = False
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12h_1d_1w_Camarilla_R1S1_Breakout_Pullback_V1"
timeframe = "6h"
leverage = 1.0