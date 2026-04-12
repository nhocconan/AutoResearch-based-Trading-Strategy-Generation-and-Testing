#!/usr/bin/env python3
"""
4h_12h_Trend_Reversal_V1
Hypothesis: On 4h timeframe, enter short when price breaks below 12h Donchian low with volume contraction and reversal candlestick (shooting star), enter long when price breaks above 12h Donchian high with volume contraction and reversal candlestick (hammer). Uses 12h Donchian channels for structure, volume contraction for exhaustion, and candlestick patterns for reversal confirmation. Designed to work in both bull and bear markets by capturing reversals at extremes with tight entry conditions to limit trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Trend_Reversal_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12H INDICATORS: Donchian channels ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = np.full_like(high_12h, np.nan)
    donchian_low = np.full_like(low_12h, np.nan)
    
    for i in range(20, len(high_12h)):
        donchian_high[i] = np.max(high_12h[i-20:i])
        donchian_low[i] = np.min(low_12h[i-20:i])
    
    # Align to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # === VOLUME CONTRACTION (exhaustion signal) ===
    vol_ma = np.zeros_like(volume)
    if len(volume) >= 20:
        vol_ma[20] = np.mean(volume[0:20])
        for i in range(21, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    volume_contraction = volume < 0.7 * vol_ma  # Volume below 70% of MA
    
    # === CANDLESTICK PATTERNS: Hammer and Shooting Star ===
    body_size = np.abs(close - open_)
    upper_shadow = high - np.maximum(open_, close)
    lower_shadow = np.minimum(open_, close) - low
    
    # Hammer: small body, long lower shadow, small upper shadow
    hammer = (body_size < 0.3 * (high - low)) & (lower_shadow > 2 * body_size) & (upper_shadow < 0.1 * (high - low))
    
    # Shooting Star: small body, long upper shadow, small lower shadow
    shooting_star = (body_size < 0.3 * (high - low)) & (upper_shadow > 2 * body_size) & (lower_shadow < 0.1 * (high - low))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if indicators not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_contraction[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions: Donchian breakout + volume contraction + reversal candle
        long_breakout = (close[i] > donchian_high_aligned[i]) and volume_contraction[i] and hammer[i]
        short_breakout = (close[i] < donchian_low_aligned[i]) and volume_contraction[i] and shooting_star[i]
        
        # Exit conditions: reversal back inside Donchian channel
        exit_long = close[i] < donchian_high_aligned[i]
        exit_short = close[i] > donchian_low_aligned[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals