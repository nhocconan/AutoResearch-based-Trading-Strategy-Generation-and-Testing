#!/usr/bin/env python3
"""
Hypothesis: 1h 4h Donchian Breakout with 1d EMA50 Trend Filter and Volume Confirmation
- Long: Price breaks above 4h Donchian upper (20) AND price > 1d EMA50 (uptrend) AND volume > 1.5x 20-period average
- Short: Price breaks below 4h Donchian lower (20) AND price < 1d EMA50 (downtrend) AND volume > 1.5x 20-period average
- Exit: Price crosses 4h Donchian midpoint (mean of upper/lower) OR trend reversal
- Uses discrete position sizing (0.20) to minimize fee churn
- Target: 15-37 trades/year (60-150 over 4 years) to avoid fee drag
- Donchian channels provide structural support/resistance; EMA50 filters for higher-timeframe trend
- Works in both bull and bear markets by only trading in direction of 1d trend
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
    
    # Get 4h data for Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Align to 1h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # Donchian needs 20, EMA50 needs 50, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Breakout conditions
        long_breakout = close[i] > donchian_upper_aligned[i]
        short_breakout = close[i] < donchian_lower_aligned[i]
        
        # Entry signals with trend filter and volume confirmation
        # Long: break above upper + uptrend + volume spike
        # Short: break below lower + downtrend + volume spike
        long_signal = long_breakout and uptrend and (volume[i] > 1.5 * vol_ma[i])
        short_signal = short_breakout and downtrend and (volume[i] > 1.5 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions: price crosses midpoint OR trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below midpoint or trend turns down
                if (close[i] < donchian_mid_aligned[i] or 
                    not uptrend):
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above midpoint or trend turns up
                if (close[i] > donchian_mid_aligned[i] or 
                    not downtrend):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Donchian20_1dEMA50_Trend_VolumeConfirm"
timeframe = "1h"
leverage = 1.0