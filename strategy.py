#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND close > 1w EMA50 AND volume > 1.5x 20-period average.
Short when price breaks below Donchian lower band AND close < 1w EMA50 AND volume > 1.5x 20-period average.
Exit when price crosses Donchian middle band (20-period mean).
Uses discrete position sizing (0.25) to balance return and drawdown. Targets 12-37 trades/year per symbol.
1w EMA50 provides strong long-term trend filter suitable for 6h timeframe.
Volume confirmation ensures breakouts have institutional participation.
Donchian channels offer clear breakout/breakdown levels with built-in exit logic.
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
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND close > 1w EMA50 AND volume spike
            if (price > donchian_upper[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND close < 1w EMA50 AND volume spike
            elif (price < donchian_lower[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Donchian middle band
            if position == 1 and price < donchian_middle[i]:
                exit_signal = True
            elif position == -1 and price > donchian_middle[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_1wEMA50_VolumeConfirm"
timeframe = "6h"
leverage = 1.0