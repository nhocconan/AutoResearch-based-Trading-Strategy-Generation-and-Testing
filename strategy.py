#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and session filter.
    # Long when price breaks above H3 and 4h close > 4h open (bullish) and hour in [8,20) UTC.
    # Short when price breaks below L3 and 4h close < 4h open (bearish) and hour in [8,20) UTC.
    # Exit when price returns to Pivot Point (PP).
    # Uses intraday breakouts filtered by higher timeframe trend to avoid false breakouts.
    # Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_ = prices['open'].values
    volume = prices['volume'].values
    
    # Calculate 1h Camarilla pivots from previous day
    # Need previous day's high, low, close
    # We'll use rolling window of 24*4=96 bars (previous day on 1h)
    if len(high) < 96:
        return np.zeros(n)
    
    # Previous day's OHLC (24 hours ago)
    prev_high = pd.Series(high).rolling(window=96, min_periods=96).max().shift(24).values
    prev_low = pd.Series(low).rolling(window=96, min_periods=96).min().shift(24).values
    prev_close = pd.Series(close).rolling(window=96, min_periods=96).last().shift(24).values
    
    # Calculate Camarilla levels
    # PP = (prev_high + prev_low + prev_close) / 3
    # H3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    # L3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    pp = (prev_high + prev_low + prev_close) / 3.0
    h3 = prev_close + 1.1 * (prev_high - prev_low) / 2.0
    l3 = prev_close - 1.1 * (prev_high - prev_low) / 2.0
    
    # Get 4h data for trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    open_4h = df_4h['open'].values
    
    # Calculate 4h trend: bullish if close > open, bearish if close < open
    bullish_4h = close_4h > open_4h
    bearish_4h = close_4h < open_4h
    
    # Align HTF indicators to 1h timeframe
    bullish_4h_aligned = align_htf_to_ltf(prices, df_4h, bullish_4h)
    bearish_4h_aligned = align_htf_to_ltf(prices, df_4h, bearish_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours < 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(96, n):  # start after we have previous day's data
        # Skip if data not ready
        if (np.isnan(pp[i]) or np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(bullish_4h_aligned[i]) or np.isnan(bearish_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            # Outside session: flatten position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > h3[i]
        short_breakout = close[i] < l3[i]
        
        # Exit condition: price returns to pivot point (PP)
        long_exit = close[i] < pp[i]  # for long position
        short_exit = close[i] > pp[i]  # for short position
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.20
        
        # Entry conditions
        if long_breakout and bullish_4h_aligned[i] and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and bearish_4h_aligned[i] and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
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

name = "1h_4h_camarilla_breakout_trend_session_v1"
timeframe = "1h"
leverage = 1.0