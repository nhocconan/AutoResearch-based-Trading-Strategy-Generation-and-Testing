#!/usr/bin/env python3

# Hypothesis: 4h Donchian breakout with volume confirmation and ADX trend filter
# Uses Donchian(20) for breakout entries, volume > 1.5x 20-period average for confirmation,
# and ADX(14) > 25 to ensure trending markets. Target: 15-35 trades/year per symbol
# with position size 0.25. Works in both bull and bear markets by capturing
# genuine breakouts with volume and trend confirmation, reducing false signals.

name = "4h_Donchian20_Volume_ADX_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Breakout conditions: price must close beyond the level
    breakout_up = close > highest_high
    breakout_down = close < lowest_low
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    # ADX trend filter (14-period)
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    plus_di = np.where(tr_sum > 0, (plus_dm_sum / tr_sum) * 100, 0)
    minus_di = np.where(tr_sum > 0, (minus_dm_sum / tr_sum) * 100, 0)
    
    dx = np.where((plus_di + minus_di) > 0, 
                  np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Trend filter: ADX > 25 indicates strong trend
    trend_filter = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(volume_filter[i]) or np.isnan(trend_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish breakout + volume + trend
            if breakout_up[i] and volume_filter[i] and trend_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + volume + trend
            elif breakout_down[i] and volume_filter[i] and trend_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian middle or trend weakens
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if close[i] <= donchian_mid or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian middle or trend weakens
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if close[i] >= donchian_mid or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals