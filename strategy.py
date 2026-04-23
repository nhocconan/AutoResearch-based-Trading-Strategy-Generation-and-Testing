#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
- Uses 1d Donchian channels (20-period high/low) for breakout signals
- 1w EMA50 defines higher timeframe trend: only trade breakouts in trend direction
- Volume confirmation (> 2.0x 20-period average) filters false breakouts
- Designed for 1d timeframe targeting 7-25 trades/year (30-100 over 4 years)
- Works in both bull and bear markets by trading with the 1w trend
- Discrete position sizing (0.0, ±0.25) to minimize fee churn
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper with 1w uptrend and volume
            long_breakout = (close[i] > high_ma[i] and 
                           close[i] > ema_50_1w_aligned[i] and
                           volume[i] > 2.0 * vol_ma[i])
            
            # Short conditions: price breaks below Donchian lower with 1w downtrend and volume
            short_breakout = (close[i] < low_ma[i] and 
                            close[i] < ema_50_1w_aligned[i] and
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
                # Exit long: price breaks below Donchian lower or 1w trend turns bearish
                if (close[i] < low_ma[i] or 
                    close[i] < ema_50_1w_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above Donchian upper or 1w trend turns bullish
                if (close[i] > high_ma[i] or 
                    close[i] > ema_50_1w_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0