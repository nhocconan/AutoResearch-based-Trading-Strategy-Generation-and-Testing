#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotTrend_1dATRStop_v1
Hypothesis: On 6h timeframe, trade long when price breaks above 20-bar Donchian high with price above 1d EMA50 trend and weekly pivot bullish bias, short when breaks below 20-bar Donchian low with price below 1d EMA50 trend and weekly pivot bearish bias. Uses volume confirmation to filter false breakouts. ATR-based stoploss (2.5*ATR) manages risk. Designed to work in both bull and bear markets by aligning with 1d trend and weekly pivot structure, reducing whipsaw in ranging markets. Targets 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize fee drag.
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
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for weekly pivot trend (bullish/bearish bias)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly pivot points and trend bias
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly bias: bullish if close > pivot, bearish if close < pivot
    weekly_bullish = close_1w > pivot_1w
    weekly_bearish = close_1w < pivot_1w
    
    # Align HTF indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # 6h Donchian channels (20-period)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # 6h ATR for stoploss and volume filter
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Volume spike: current volume > 1.8 * 30-period average (stricter for 6h)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA50 (50), Donchian (20), ATR (14), volume MA (30)
    start_idx = max(50, 20, 14, 30) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_val = ema_50_1d_aligned[i]
        weekly_bull = weekly_bullish_aligned[i] > 0.5
        weekly_bear = weekly_bearish_aligned[i] > 0.5
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        close_val = close[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above Donchian high, above 1d EMA50, weekly bullish, with volume spike
            long_signal = (close_val > donchian_high) and (close_val > ema_50_val) and weekly_bull and vol_spike
            
            # Short: price breaks below Donchian low, below 1d EMA50, weekly bearish, with volume spike
            short_signal = (close_val < donchian_low) and (close_val < ema_50_val) and weekly_bear and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below Donchian low OR ATR stoploss (2.5*ATR below entry)
            if (close_val < donchian_low) or (close_val < entry_price - 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above Donchian high OR ATR stoploss (2.5*ATR above entry)
            if (close_val > donchian_high) or (close_val > entry_price + 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotTrend_1dATRStop_v1"
timeframe = "6h"
leverage = 1.0