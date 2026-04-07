#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d Trend and Volume Confirmation
Breakouts above/below 20-period Donchian channels with 1d trend filter and volume confirmation.
Trades only in direction of higher timeframe trend to reduce whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d Trend Filter (Higher Timeframe) ===
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Donchian Channels (20-period) ===
    # Calculate rolling max/min
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume Confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any data is NaN
        if np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below lower Donchian band
            if close[i] < lowest_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above upper Donchian band
            if close[i] > highest_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry conditions with 1d trend filter
            if close[i] > highest_20[i] and close[i] > ema_50_1d_aligned[i]:
                # Breakout above upper band with 1d uptrend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < lowest_20[i] and close[i] < ema_50_1d_aligned[i]:
                # Breakdown below lower band with 1d downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals