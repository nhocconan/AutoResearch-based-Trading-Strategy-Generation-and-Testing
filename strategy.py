#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_WeeklyTrend_ATRStop_v1
Hypothesis: Daily Donchian(20) breakouts in direction of weekly EMA50 trend with ATR-based trailing stop capture major trends while avoiding whipsaw. Uses discrete sizing (0.25) targeting ~15 trades/year. Works in bull/bear by taking breakouts aligned with weekly trend. Weekly timeframe ensures we only trade with the dominant multi-week momentum, reducing false breakouts in chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load weekly data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # ATR(14) for stoploss calculation
    tr1 = pd.Series(high[1:] - low[1:]).values
    tr2 = pd.Series(np.abs(high[1:] - close[:-1])).values
    tr3 = pd.Series(np.abs(low[1:] - close[:-1])).values
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of Donchian (20), EMA50 (50), ATR (14)
    start_idx = max(20, 50, 14)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        trend_val = ema50_1w_aligned[i]
        atr_val = atr[i]
        
        # Skip if any data not ready
        if np.isnan(trend_val) or np.isnan(atr_val):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Calculate Donchian channels for previous 20 periods
        if i >= 20:
            donchian_high = np.max(high[i-20:i])
            donchian_low = np.min(low[i-20:i])
        else:
            donchian_high = high_val
            donchian_low = low_val
        
        # Trend filter: price > weekly EMA50 = uptrend, price < weekly EMA50 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Donchian breakout conditions
        long_breakout = close_val > donchian_high
        short_breakout = close_val < donchian_low
        
        # Entry conditions: Donchian breakout in direction of weekly trend
        long_entry = long_breakout and is_uptrend
        short_entry = short_breakout and is_downtrend
        
        # Update highest/lowest for trailing stop (ATR-based)
        if position == 1:
            highest_since_long = max(highest_since_long, high_val)
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low_val)
        elif position == 0:
            highest_since_long = 0.0
            lowest_since_short = 0.0
        
        # Exit conditions: ATR-based trailing stoploss
        long_exit = False
        short_exit = False
        if position == 1:
            # Long trailing stop: highest since entry - 2.5 * ATR
            stop_price = highest_since_long - 2.5 * atr_val
            long_exit = close_val < stop_price
        elif position == -1:
            # Short trailing stop: lowest since entry + 2.5 * ATR
            stop_price = lowest_since_short + 2.5 * atr_val
            short_exit = close_val > stop_price
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            highest_since_long = high_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            lowest_since_short = low_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
        elif short_exit:
            signals[i] = 0.0
            position = 0
            lowest_since_short = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyTrend_ATRStop_v1"
timeframe = "1d"
leverage = 1.0