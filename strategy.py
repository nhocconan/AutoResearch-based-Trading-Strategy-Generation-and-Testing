#!/usr/bin/env python3
"""
4h_12h_HeikinAshi_Trend_Follower_v1
Hypothesis: On 4h timeframe, use Heikin Ashi candles from 12h timeframe to identify strong trends.
Go long when 12h Heikin Ashi shows three consecutive bullish candles with no lower shadows.
Go short when 12h Heikin Ashi shows three consecutive bearish candles with no upper shadows.
Exit when trend weakens (opposite color candle appears).
Uses volume confirmation on 4h to avoid false signals.
Designed to catch strong trends in both bull and bear markets while avoiding choppy periods.
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
    
    # Load 12h data for Heikin Ashi calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    open_12h = df_12h['open'].values
    close_12h = df_12h['close'].values
    
    # Calculate Heikin Ashi candles
    ha_close = np.zeros_like(close_12h)
    ha_open = np.zeros_like(close_12h)
    ha_high = np.zeros_like(close_12h)
    ha_low = np.zeros_like(close_12h)
    
    for i in range(len(close_12h)):
        if i == 0:
            ha_close[i] = (open_12h[i] + high_12h[i] + low_12h[i] + close_12h[i]) / 4
            ha_open[i] = (open_12h[i] + close_12h[i]) / 2
        else:
            ha_close[i] = (open_12h[i] + high_12h[i] + low_12h[i] + close_12h[i]) / 4
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
        ha_high[i] = max(high_12h[i], ha_open[i], ha_close[i])
        ha_low[i] = min(low_12h[i], ha_open[i], ha_close[i])
    
    # Identify trend strength: consecutive same-color candles with no opposing shadows
    bullish_streak = np.zeros_like(ha_close, dtype=int)
    bearish_streak = np.zeros_like(ha_close, dtype=int)
    
    for i in range(len(ha_close)):
        if ha_close[i] >= ha_open[i]:  # Bullish candle
            bullish_streak[i] = bullish_streak[i-1] + 1 if i > 0 else 1
            bearish_streak[i] = 0
        else:  # Bearish candle
            bearish_streak[i] = bearish_streak[i-1] + 1 if i > 0 else 1
            bullish_streak[i] = 0
    
    # Strong trend conditions: 3+ consecutive candles with no opposing shadows
    strong_bullish = np.zeros_like(ha_close, dtype=bool)
    strong_bearish = np.zeros_like(ha_close, dtype=bool)
    
    for i in range(len(ha_close)):
        if bullish_streak[i] >= 3 and ha_low[i] >= ha_open[i]:  # No lower shadow
            strong_bullish[i] = True
        if bearish_streak[i] >= 3 and ha_high[i] <= ha_open[i]:  # No upper shadow
            strong_bearish[i] = True
    
    # Align trend signals to 4h timeframe
    strong_bullish_aligned = align_htf_to_ltf(prices, df_12h, strong_bullish.astype(float))
    strong_bearish_aligned = align_htf_to_ltf(prices, df_12h, strong_bearish.astype(float))
    
    # Volume confirmation on 4h: volume > 1.5x 20-period average
    vol_ma_20 = np.full_like(volume, np.nan)
    for j in range(19, len(volume)):
        vol_ma_20[j] = np.mean(volume[j-19:j+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(strong_bullish_aligned[i]) or np.isnan(strong_bearish_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Look for long entry: strong bullish trend on 12h with volume confirmation
            if strong_bullish_aligned[i] > 0.5 and volume_ratio > 1.5:
                position = 1
                signals[i] = position_size
            # Look for short entry: strong bearish trend on 12h with volume confirmation
            elif strong_bearish_aligned[i] > 0.5 and volume_ratio > 1.5:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend weakens (bearish signal appears) or volume drops
            if strong_bearish_aligned[i] > 0.5 or volume_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: trend weakens (bullish signal appears) or volume drops
            if strong_bullish_aligned[i] > 0.5 or volume_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_HeikinAshi_Trend_Follower_v1"
timeframe = "4h"
leverage = 1.0