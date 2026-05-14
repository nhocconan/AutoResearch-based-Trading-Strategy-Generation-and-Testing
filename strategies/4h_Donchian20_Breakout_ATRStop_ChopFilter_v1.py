#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_ATRStop_ChopFilter_v1
Hypothesis: Donchian(20) breakouts on 4h timeframe with ATR-based stoploss and choppiness regime filter capture strong trending moves while avoiding whipsaw in ranging markets. Uses discrete sizing (0.30) to target 20-50 trades/year. Works in bull/bear by taking breakouts in direction of higher-timeframe trend (1d EMA50). Volume confirmation (>1.5x 20-bar average) ensures momentum validity. Choppiness Index > 61.8 avoids range-bound false breakouts.
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # ATR(14) for stoploss calculation
    tr1 = pd.Series(high[1:] - low[1:]).values
    tr2 = pd.Series(np.abs(high[1:] - close[:-1])).values
    tr3 = pd.Series(np.abs(low[1:] - close[:-1])).values
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    # Choppiness Index (14) for regime filter
    def calculate_chop(high, low, close, window=14):
        atr_sum = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=window, min_periods=window).sum()
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max()
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min()
        chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(window)
        return chop.values
    
    chop = calculate_chop(high, low, close, 14)
    chop_filter = chop > 61.8  # Only trade when choppy/range-bound (mean reversion context) OR actually: we want to avoid chop for breakouts, so chop < 38.2 for trending
    # Correction: For breakout strategies, we want trending markets (low chop), not choppy
    chop_filter = chop < 38.2  # Only trade when market is trending (low choppiness)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    entry_price = 0.0
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of Donchian (20), EMA50 (50), ATR (14), volume MA (20), chop (14)
    start_idx = max(20, 50, 14, 20, 14)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        trend_val = ema50_1d_aligned[i]
        atr_val = atr[i]
        vol_conf = volume_confirm[i]
        chop_val = chop[i]
        is_trending = chop_filter[i]
        
        # Skip if any data not ready
        if (np.isnan(trend_val) or np.isnan(atr_val) or np.isnan(chop_val)):
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
        
        # Trend filter: price > 1d EMA50 = uptrend, price < 1d EMA50 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Donchian breakout conditions
        long_breakout = close_val > donchian_high
        short_breakout = close_val < donchian_low
        
        # Entry conditions: Donchian breakout in direction of 1d trend + volume + trending regime
        long_entry = long_breakout and is_uptrend and vol_conf and is_trending
        short_entry = short_breakout and is_downtrend and vol_conf and is_trending
        
        # Update highest/lowest for trailing stop (ATR-based)
        if position == 1:
            highest_since_long = max(highest_since_long, high_val)
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low_val)
        elif position == 0:
            highest_since_long = 0.0
            lowest_since_short = 0.0
        
        # Exit conditions: ATR-based trailing stoploss or opposite Donchian touch
        long_exit = False
        short_exit = False
        if position == 1:
            # Long trailing stop: highest since entry - 2.5 * ATR
            stop_price = highest_since_long - 2.5 * atr_val
            long_exit = close_val < stop_price or close_val < donchian_low  # Stop or Donchian breakdown
        elif position == -1:
            # Short trailing stop: lowest since entry + 2.5 * ATR
            stop_price = lowest_since_short + 2.5 * atr_val
            short_exit = close_val > stop_price or close_val > donchian_high  # Stop or Donchian breakout
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
            highest_since_long = high_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
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

name = "4h_Donchian20_Breakout_ATRStop_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0