#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
- Long: Close breaks above Donchian(20) high + price > 1d EMA34 + volume > 2.0x 20-period average
- Short: Close breaks below Donchian(20) low + price < 1d EMA34 + volume > 2.0x 20-period average
- Exit: Close crosses 1d EMA34 (trend reversal)
- Uses discrete position sizing (0.30) to balance return and risk
- Target: 12-30 trades/year (50-120 over 4 years) to avoid fee drag
- Donchian channels provide clear breakout levels; EMA34 filters for primary trend
- Works in both bull and bear markets by only taking trades in direction of higher timeframe trend
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Donchian(20) channels
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, donchian_window, 20)  # EMA34 needs 34, Donchian needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA34
        uptrend = close[i] > ema34_aligned[i]
        downtrend = close[i] < ema34_aligned[i]
        
        # Donchian breakout signals with trend filter and volume confirmation
        # Long: Close breaks above Donchian high + uptrend + volume spike
        # Short: Close breaks below Donchian low + downtrend + volume spike
        long_signal = (close[i] > donchian_high[i] and 
                      uptrend and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (close[i] < donchian_low[i] and 
                       downtrend and
                       volume[i] > 2.0 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
        else:
            # Exit: Close crosses 1d EMA34 (trend reversal)
            exit_signal = False
            
            if position == 1:
                # Exit long: Close crosses below EMA34 (trend turns down)
                if close[i] < ema34_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Close crosses above EMA34 (trend turns up)
                if close[i] > ema34_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "12h_Donchian20_1dEMA34_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0