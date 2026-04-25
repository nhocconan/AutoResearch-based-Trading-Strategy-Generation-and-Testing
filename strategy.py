#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dEMA34_Trend_VolumeConfirmation
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation. Uses HTF 1d EMA for trend alignment and 4h volume spike (>2.0x 20-bar mean) for entry confirmation. Designed to capture strong trending moves in both bull and bear markets with tight entries to minimize fee drift. Target: 20-30 trades/year per symbol.
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
    
    # Get 1d data for HTF trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) channels on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0x 20-bar mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian and volume mean
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or 
            np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_mean_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above Donchian(20) high in uptrend (close > EMA34) with volume confirmation
            # Short: price breaks below Donchian(20) low in downtrend (close < EMA34) with volume confirmation
            long_signal = (close[i] > highest_20[i]) and (close[i] > ema_34_aligned[i]) and vol_confirm[i]
            short_signal = (close[i] < lowest_20[i]) and (close[i] < ema_34_aligned[i]) and vol_confirm[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below Donchian(20) low (mean reversion or trend end)
            exit_signal = close[i] < lowest_20[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Donchian(20) high (mean reversion or trend end)
            exit_signal = close[i] > highest_20[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0