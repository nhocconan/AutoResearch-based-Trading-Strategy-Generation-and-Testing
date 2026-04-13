#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and session filter
    # Long when: price breaks above H3 (bullish breakout) AND 4h close > 4h open (uptrend) AND 08-20 UTC
    # Short when: price breaks below L3 (bearish breakout) AND 4h close < 4h open (downtrend) AND 08-20 UTC
    # Exit when: price returns to pivot point (mean reversion) OR adverse 4h trend flip
    # Uses discrete sizing (0.20) targeting 60-150 trades over 4 years.
    # Works in bull/bear via 4h trend filter preventing counter-trend trades.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    open_4h = df_4h['open'].values
    
    # Calculate 4h trend: bullish if close > open, bearish if close < open
    trend_4h_bullish = close_4h > open_4h
    trend_4h_bearish = close_4h < open_4h
    trend_4h_bullish_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_bullish.astype(float))
    trend_4h_bearish_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_bearish.astype(float))
    
    # Calculate Camarilla pivots for 1h using previous day's OHLC
    # Get daily OHLC from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Previous day's OHLC (shifted by 1 to avoid look-ahead)
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_open_1d = df_1d['open'].shift(1).values
    
    # Calculate Camarilla levels
    range_1d = prev_high_1d - prev_low_1d
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    h3 = pivot + (range_1d * 1.1 / 4)
    l3 = pivot - (range_1d * 1.1 / 4)
    h4 = pivot + (range_1d * 1.1 / 2)
    l4 = pivot - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 1h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(pivot_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(trend_4h_bullish_aligned[i]) or np.isnan(trend_4h_bearish_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        bullish_breakout = close[i] > h3_aligned[i]
        bearish_breakout = close[i] < l3_aligned[i]
        
        # Mean reversion exit (return to pivot)
        return_to_pivot = abs(close[i] - pivot_aligned[i]) < (h3_aligned[i] - l3_aligned[i]) * 0.1
        
        # 4h trend conditions
        uptrend_4h = trend_4h_bullish_aligned[i] > 0.5
        downtrend_4h = trend_4h_bearish_aligned[i] > 0.5
        
        # Entry conditions
        long_entry = bullish_breakout and uptrend_4h and position != 1
        short_entry = bearish_breakout and downtrend_4h and position != -1
        
        # Exit conditions
        exit_long = return_to_pivot or (position == 1 and not uptrend_4h)
        exit_short = return_to_pivot or (position == -1 and not downtrend_4h)
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
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

name = "1h_4h_1d_camarilla_breakout_trend_filter_session_v1"
timeframe = "1h"
leverage = 1.0