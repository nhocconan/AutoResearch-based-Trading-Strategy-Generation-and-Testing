#!/usr/bin/env python3
"""
6h_WeeklySwingRejection
Hypothesis: On the 6h timeframe, price often rejects at weekly swing highs/lows during ranging markets (2025+). 
We identify weekly swing points (highest high/lowest low over 4 weeks) and enter short when price rejects 
from weekly resistance with bearish engulfing, and long when price rejects from weekly support with bullish engulfing.
Uses 1d volatility filter (ATR ratio) to avoid whipsaws in high volatility. Targets 50-150 total trades over 4 years.
Works in both bull/bear as it fades extremes rather than following trends.
"""

name = "6h_WeeklySwingRejection"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for swing points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    open_6h = prices['open'].values
    
    # --- 1d ATR for volatility filter (14 period) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1d / (atr_ma_1d + 1e-10)
    atr_ratio_6h = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # --- Weekly Swing Points (4-week lookback) ---
    # Swing high: highest high over past 4 weeks
    # Swing low: lowest low over past 4 weeks
    week_high = df_1w['high'].values
    week_low = df_1w['low'].values
    
    swing_high = pd.Series(week_high).rolling(window=4, min_periods=4).max().values
    swing_low = pd.Series(week_low).rolling(window=4, min_periods=4).min().values
    
    # Align to 6h
    swing_high_6h = align_htf_to_ltf(prices, df_1w, swing_high)
    swing_low_6h = align_htf_to_ltf(prices, df_1w, swing_low)
    
    # --- 6h Engulfing Patterns ---
    # Bullish engulfing: current green candle engulfs previous red candle
    # Bearish engulfing: current red candle engulfs previous green candle
    bull_engulf = (close_6h > open_6h) & (open_6h > np.roll(close_6h, 1)) & (close_6h > np.roll(open_6h, 1))
    bear_engulf = (close_6h < open_6h) & (open_6h < np.roll(close_6h, 1)) & (close_6h < np.roll(open_6h, 1))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(atr_ratio_6h[i]) or np.isnan(swing_high_6h[i]) or 
            np.isnan(swing_low_6h[i])):
            if position != 0:
                # Emergency exit: reverse signal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: only trade in low-moderate volatility
        # Avoid whipsaws in high volatility environments
        vol_filter = atr_ratio_6h[i] < 1.5
        
        if position == 0 and vol_filter:
            # Look for rejection at weekly swing levels
            # Long setup: bullish engulfing at or above weekly swing low
            if bull_engulf[i] and low_6h[i] <= swing_low_6h[i] * 1.001:  # near swing low
                signals[i] = 0.25
                position = 1
                entry_price = close_6h[i]
            # Short setup: bearish engulfing at or below weekly swing high
            elif bear_engulf[i] and high_6h[i] >= swing_high_6h[i] * 0.999:  # near swing high
                signals[i] = -0.25
                position = -1
                entry_price = close_6h[i]
        
        elif position == 1:
            # Long position management
            # Take profit: price reaches weekly swing high or shows bearish engulfing resistance
            if high_6h[i] >= swing_high_6h[i] * 0.999 or bear_engulf[i]:
                signals[i] = 0.0
                position = 0
            # Stop loss: price breaks below weekly swing low with bearish engulfing
            elif low_6h[i] < swing_low_6h[i] * 0.999 and bear_engulf[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short position management
            # Take profit: price reaches weekly swing low or shows bullish engulfing support
            if low_6h[i] <= swing_low_6h[i] * 1.001 or bull_engulf[i]:
                signals[i] = 0.0
                position = 0
            # Stop loss: price breaks above weekly swing high with bullish engulfing
            elif high_6h[i] > swing_high_6h[i] * 1.001 and bull_engulf[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals