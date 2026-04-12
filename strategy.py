#!/usr/bin/env python3
"""
12h_1d_donchian_breakout_volume_filter
Hypothesis: 12-hour strategy using daily Donchian channel breakouts with volume confirmation to capture trends in both bull and bear markets.
Uses daily Donchian(20) breakouts confirmed by volume > 1.5x 20-period average.
Breakouts in direction of daily EMA50 trend to avoid counter-trend trades.
Position sizing fixed at 0.25 to minimize churn. Target: 15-25 trades/year.
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
    
    # Get daily data for trend and breakout levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily Donchian(20) channels
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily volume average for confirmation
    vol_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend determination
        uptrend = ema50_1d_aligned[i] > close_1d[min(i//2, len(close_1d)-1)] if i//2 < len(close_1d) else ema50_1d_aligned[i] > ema50_1d_aligned[i-1]
        downtrend = ema50_1d_aligned[i] < close_1d[min(i//2, len(close_1d)-1)] if i//2 < len(close_1d) else ema50_1d_aligned[i] < ema50_1d_aligned[i-1]
        
        # Volume confirmation: current 12h volume > 1.5x daily average volume
        vol_confirm = volume[i] > (vol_avg_aligned[i] * 1.5)
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high_aligned[i]
        short_breakout = close[i] < donchian_low_aligned[i]
        
        # Entry logic
        if uptrend and long_breakout and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        elif downtrend and short_breakout and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout or trend reversal
        elif position == 1 and (short_breakout or (downtrend and close[i] < ema50_1d_aligned[i])):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (long_breakout or (uptrend and close[i] > ema50_1d_aligned[i])):
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_donchian_breakout_volume_filter"
timeframe = "12h"
leverage = 1.0