#!/usr/bin/env python3
"""
4h_1d_donchian_breakout_volume_v2
Hypothesis: On 4h timeframe, price breaking above Donchian(20) high or below Donchian(20) low with daily volume expansion and ATR-based stoploss captures breakout moves. The strategy is designed to work in both bull and bear markets by focusing on breakout strength rather than direction bias. Volume expansion filters false breakouts, and ATR stoploss manages risk. Target: 20-50 trades/year to avoid excessive fee drag.
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
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        donchian_high[i] = np.max(high[i - lookback + 1:i + 1])
        donchian_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Calculate daily volume expansion: current volume > 1.5x 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume_1d > (vol_ma_20 * 1.5)
    
    # Calculate ATR (14-period) for stoploss
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily volume expansion to 4h timeframe
    volume_expansion_aligned = align_htf_to_ltf(prices, df_1d, volume_expansion.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_expansion_aligned[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Break of Donchian channel with volume expansion
        long_break = close[i] > donchian_high[i]
        short_break = close[i] < donchian_low[i]
        
        long_entry = long_break and volume_expansion_aligned[i] > 0.5
        short_entry = short_break and volume_expansion_aligned[i] > 0.5
        
        # Stoploss conditions: ATR-based
        stop_long = position == 1 and close[i] <= donchian_high[i - 1] - 2.0 * atr[i]
        stop_short = position == -1 and close[i] >= donchian_low[i - 1] + 2.0 * atr[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif stop_long or stop_short:
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

name = "4h_1d_donchian_breakout_volume_v2"
timeframe = "4h"
leverage = 1.0