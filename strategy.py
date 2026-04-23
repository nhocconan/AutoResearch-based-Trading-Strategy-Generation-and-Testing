#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
- Long: Close > Donchian High(20) AND price > 1d EMA34 (uptrend) AND volume > 2.0x 20-period average
- Short: Close < Donchian Low(20) AND price < 1d EMA34 (downtrend) AND volume > 2.0x 20-period average
- Exit: Opposite Donchian breakout (close crosses middle line) OR trend reversal
- Uses discrete position sizing (0.25) to minimize fee churn
- Target: 12-30 trades/year (50-120 over 4 years) to avoid fee drag
- Donchian channels provide clear structure; EMA34 filters for higher timeframe trend
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
    
    # Calculate Donchian(20) channels on 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    donchian_high = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 20)  # EMA34 needs 34, Donchian needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA34
        uptrend = close[i] > ema34_aligned[i]
        downtrend = close[i] < ema34_aligned[i]
        
        # Donchian breakout signals with trend filter and volume confirmation
        # Long: Close > Donchian High(20) + uptrend + volume spike
        # Short: Close < Donchian Low(20) + downtrend + volume spike
        long_signal = (close[i] > donchian_high_aligned[i] and 
                      uptrend and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (close[i] < donchian_low_aligned[i] and 
                       downtrend and
                       volume[i] > 2.0 * vol_ma[i])
        
        # Exit conditions: Opposite Donchian breakout OR trend reversal
        exit_signal = False
        if position == 1:
            # Exit long: Close < Donchian Mid OR trend turns down
            if (close[i] < donchian_mid_aligned[i] or 
                not uptrend):
                exit_signal = True
        elif position == -1:
            # Exit short: Close > Donchian Mid OR trend turns up
            if (close[i] > donchian_mid_aligned[i] or 
                not downtrend):
                exit_signal = True
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0