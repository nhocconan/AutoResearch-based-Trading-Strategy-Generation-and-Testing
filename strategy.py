#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wTrend_RegimeFilter_v1
Hypothesis: Trade 12h Donchian(20) breakouts with 1-week EMA50 trend filter and choppiness regime filter (CHOP < 50).
Only trade when 1w trend is aligned (price > EMA50 for long, price < EMA50 for short) AND market is trending (Choppiness Index < 50).
Exit on opposite Donchian touch or trend reversal.
Position size: 0.25 to balance profit and fee drag.
Target: 12-37 trades/year (50-150 over 4 years) to stay within 12h hard max of 200 total trades.
Works in bull (breakouts with trend) and bear (strong breakdowns with trend) markets by filtering for aligned HTF trend and trending regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h Donchian channels (20-period)
    donch_period = 20
    highest_high = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    lowest_low = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Calculate 12h Choppiness Index for regime filter (trending when CHOP < 50)
    chop_period = 14
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    highest_high_chop = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low_chop = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    hl_range = highest_high_chop - lowest_low_chop
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    chop = 100 * np.log10(pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values / hl_range) / np.log10(chop_period)
    chop = np.where(np.isnan(chop), 50.0, chop)  # default to neutral if not enough data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20), EMA50 (50), ATR (14), and CHOP (14)
    start_idx = max(donch_period, 50, chop_period)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend (bullish = price above EMA50)
        htf_1w_bullish = close[i] > ema_50_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_50_1w_aligned[i]
        
        # Regime filter: only trade in trending markets (CHOP < 50)
        is_trending = chop[i] < 50.0
        
        if position == 0:
            # Long setup: price breaks above Donchian upper + 1w uptrend + trending regime
            long_setup = (close[i] > highest_high[i]) and htf_1w_bullish and is_trending
            
            # Short setup: price breaks below Donchian lower + 1w downtrend + trending regime
            short_setup = (close[i] < lowest_low[i]) and htf_1w_bearish and is_trending
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches Donchian lower (stop) OR 1w trend turns bearish OR regime turns choppy
            if (close[i] <= lowest_low[i]) or (not htf_1w_bullish) or (not is_trending):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Donchian upper (stop) OR 1w trend turns bullish OR regime turns choppy
            if (close[i] >= highest_high[i]) or (htf_1w_bullish) or (not is_trending):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1wTrend_RegimeFilter_v1"
timeframe = "12h"
leverage = 1.0