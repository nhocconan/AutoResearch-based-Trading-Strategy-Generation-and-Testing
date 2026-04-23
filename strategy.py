#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
- Donchian(20) on 12h provides clear breakout levels with proven edge in trending markets
- 1d EMA50 defines higher timeframe trend: only trade breakouts in trend direction
- Volume confirmation (> 2.0x 20-period average) filters false breakouts
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading with the 1d trend
- Discrete position sizing (0.25) minimizes fee churn
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
    
    # Calculate 12h Donchian(20) channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper/lower: 20-period high/low
    donchian_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # for EMA50 and Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high with 1d uptrend and volume
            long_breakout = (close[i] > donchian_high_aligned[i] and 
                           close[i] > ema_50_1d_aligned[i] and
                           volume[i] > 2.0 * vol_ma[i])
            
            # Short conditions: price breaks below Donchian low with 1d downtrend and volume
            short_breakout = (close[i] < donchian_low_aligned[i] and 
                            close[i] < ema_50_1d_aligned[i] and
                            volume[i] > 2.0 * vol_ma[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Donchian breakout or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian low or 1d trend turns bearish
                if (close[i] < donchian_low_aligned[i] or 
                    close[i] < ema_50_1d_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above Donchian high or 1d trend turns bullish
                if (close[i] > donchian_high_aligned[i] or 
                    close[i] > ema_50_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0