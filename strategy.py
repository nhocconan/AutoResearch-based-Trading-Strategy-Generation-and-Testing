#!/usr/bin/env python3
"""
1d_1w_Weekly_Candlestick_Reversal
Hypothesis: Uses weekly candlestick patterns (engulfing, pin bar) at key support/resistance levels identified on weekly timeframe.
Enters on daily close confirmation of the weekly pattern with volume filter.
Works in both bull and bear markets by capturing reversals at extreme weekly levels.
Target: 10-25 trades/year on 1d (40-100 total over 4 years).
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
    volume = prices['volume'].values
    
    # Get weekly data for pattern detection and levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly candlestick patterns
    body_1w = np.abs(close_1w - open_1w) if 'open' in df_1w.columns else np.abs(close_1w - np.roll(close_1w, 1))
    # Since we don't have open in 1w data from get_htf_data, approximate using close
    open_1w = np.roll(close_1w, 1)
    open_1w[0] = close_1w[0]  # first value
    
    body_size = np.abs(close_1w - open_1w)
    upper_wick = high_1w - np.maximum(close_1w, open_1w)
    lower_wick = np.minimum(close_1w, open_1w) - low_1w
    
    # Bullish engulfing: current green candle engulfs previous red candle
    bullish_engulfing = (close_1w > open_1w) & (open_1w > np.roll(close_1w, 1)) & \
                        (np.roll(open_1w, 1) > np.roll(close_1w, 2)) & \
                        (body_size > np.roll(body_size, 1))
    
    # Bearish engulfing: current red candle engulfs previous green candle
    bearish_engulfing = (close_1w < open_1w) & (open_1w < np.roll(close_1w, 1)) & \
                        (np.roll(open_1w, 1) < np.roll(close_1w, 2)) & \
                        (body_size > np.roll(body_size, 1))
    
    # Bullish pin bar: long lower wick, small body, small upper wick
    bullish_pin = (lower_wick > 2 * body_size) & (upper_wick < 0.5 * body_size) & (body_size > 0)
    
    # Bearish pin bar: long upper wick, small body, small lower wick
    bearish_pin = (upper_wick > 2 * body_size) & (lower_wick < 0.5 * body_size) & (body_size > 0)
    
    # Weekly support/resistance levels: recent highs/lows
    # Support: lowest low of last 4 weeks
    resistance = pd.Series(high_1w).rolling(window=4, min_periods=4).max().values
    support = pd.Series(low_1w).rolling(window=4, min_periods=4).min().values
    
    # Price near support/resistance (within 1% of level)
    near_support = np.minimum(np.abs(high_1w - support), np.abs(low_1w - support)) < (0.01 * support)
    near_resistance = np.minimum(np.abs(high_1w - resistance), np.abs(low_1w - resistance)) < (0.01 * resistance)
    
    # Combine patterns with levels
    bullish_setup = (bullish_engulfing | bullish_pin) & near_support
    bearish_setup = (bearish_engulfing | bearish_pin) & near_resistance
    
    # Align to daily timeframe
    bullish_setup_aligned = align_htf_to_ltf(prices, df_1w, bullish_setup)
    bearish_setup_aligned = align_htf_to_ltf(prices, df_1w, bearish_setup)
    support_aligned = align_htf_to_ltf(prices, df_1w, support)
    resistance_aligned = align_htf_to_ltf(prices, df_1w, resistance)
    
    # Daily volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not session_mask[i] or \
           np.isnan(bullish_setup_aligned[i]) or \
           np.isnan(bearish_setup_aligned[i]) or \
           np.isnan(support_aligned[i]) or \
           np.isnan(resistance_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation on daily
        vol_ma_20_today = vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else 0
        volume_expansion_today = volume[i] > (vol_ma_20_today * 1.5) if vol_ma_20_today > 0 else False
        
        # Long entry: bullish weekly setup + price near support + volume expansion
        if bullish_setup_aligned[i] and close[i] > support_aligned[i] and volume_expansion_today:
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        # Short entry: bearish weekly setup + price near resistance + volume expansion
        elif bearish_setup_aligned[i] and close[i] < resistance_aligned[i] and volume_expansion_today:
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        # Exit conditions: opposite setup appears
        elif bearish_setup_aligned[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif bullish_setup_aligned[i] and position == -1:
            position = 0
            signals[i] = 0.0
        # Hold position
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_1w_Weekly_Candlestick_Reversal"
timeframe = "1d"
leverage = 1.0