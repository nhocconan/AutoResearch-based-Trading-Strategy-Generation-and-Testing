#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
- Donchian(20) provides clear breakout levels from recent price structure
- 1d EMA34 defines intermediate trend: only long when price > EMA34, short when price < EMA34
- Volume confirmation (> 1.5x 24-period MA) reduces false breakouts in low-volume environments
- Designed for 12h timeframe to capture medium-term swings with controlled frequency (target: 12-37 trades/year)
- Works in bull via long breakouts above upper channel and in bear via short breakdowns below lower channel
- Uses discrete position sizing (0.25) to minimize fee churn
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
    
    # Calculate 1d Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian levels: upper = max(high, 20), lower = min(low, 20)
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe (use previous day's levels for breakout)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 1.5x 24-period average (12 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 24)  # need Donchian(20), EMA34, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian AND above 1d EMA34 AND volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND below 1d EMA34 AND volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level OR crosses 1d EMA34
            exit_signal = False
            
            if position == 1:
                # Exit long when price < lower Donchian OR < 1d EMA34
                if close[i] < donchian_lower_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > upper Donchian OR > 1d EMA34
                if close[i] > donchian_upper_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0