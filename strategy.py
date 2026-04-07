#!/usr/bin/env python3
"""
4h_donchian_breakout_volume_v2
Hypothesis: Donchian channel breakouts with volume confirmation provide edge in both bull and bear markets.
Long when price breaks above 20-period Donchian high with volume > 1.5x average, short when breaks below
Donchian low with volume confirmation. Use 1-day trend filter to avoid counter-trend trades.
Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    def calculate_donchian(high, low, window):
        upper = np.full(len(high), np.nan)
        lower = np.full(len(high), np.nan)
        for i in range(window-1, len(high)):
            upper[i] = np.max(high[i-window+1:i+1])
            lower[i] = np.min(low[i-window+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50d = pd.Series(close_1d).ewm(span=50, min_periods=50).mean().values
    ema_50d_aligned = align_htf_to_ltf(prices, df_1d, ema_50d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or np.isnan(ema_50d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Conditions
        volume_confirmed = volume[i] > (vol_ma[i] * 1.5)
        bullish_trend = ema_50d_aligned[i] > ema_50d_aligned[i-1] if i > 0 else False
        bearish_trend = ema_50d_aligned[i] < ema_50d_aligned[i-1] if i > 0 else False
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower or trend reverses
            if close[i] < donchian_lower[i] or (bullish_trend == False and bearish_trend == True):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper or trend reverses
            if close[i] > donchian_upper[i] or (bullish_trend == True and bearish_trend == False):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: break above Donchian upper with volume confirmation and bullish daily trend
            if close[i] > donchian_upper[i] and volume_confirmed and bullish_trend:
                position = 1
                signals[i] = 0.25
            # Short: break below Donchian lower with volume confirmation and bearish daily trend
            elif close[i] < donchian_lower[i] and volume_confirmed and bearish_trend:
                position = -1
                signals[i] = -0.25
    
    return signals