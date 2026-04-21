#!/usr/bin/env python3
"""
4h_ADXTrend_VolumeSpike_DonchianExit_V1
Hypothesis: ADX > 25 identifies strong trends, volume spike (>2x 20-bar MA) confirms momentum, and Donchian(20) breakout in trend direction captures moves. Exit on opposite Donchian breakout. Works in both bull and bear markets by catching strong trending moves while avoiding chop. Target: 30-60 trades/year per symbol (120-240 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Calculate ADX on primary timeframe (4h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = down_move[0] = np.nan
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume filter: 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation (>2x average)
        volume_ok = volume > 2.0 * vol_ma[i]
        
        # Strong trend filter (ADX > 25)
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Long: bullish DI crossover + volume + strong trend
            if plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1]:
                if volume_ok and strong_trend:
                    signals[i] = 0.30
                    position = 1
            # Short: bearish DI crossover + volume + strong trend
            elif minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1]:
                if volume_ok and strong_trend:
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:
            # Exit: price breaks below Donchian low (opposite breakout)
            if price < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit: price breaks above Donchian high (opposite breakout)
            if price > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_ADXTrend_VolumeSpike_DonchianExit_V1"
timeframe = "4h"
leverage = 1.0