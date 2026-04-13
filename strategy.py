#!/usr/bin/env python3
"""
12h_1d_1w_Triple_Timeframe_Donchian
Hypothesis: Combines weekly trend filter, daily Donchian channel breakout, and 12h precise entry with volume confirmation.
Weekly trend (price above/below 200 SMA) filters direction, daily Donchian(20) breakout provides entry signals,
and 12h volume confirmation ensures conviction. Designed for low-frequency, high-conviction trades that work in both bull and bear markets by following the major trend.
Target: 12-37 trades/year on 12h (50-150 total over 4 years).
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly 200-period SMA for trend filter
    sma_200_1w = pd.Series(close_1w).rolling(window=200, min_periods=200).mean()
    # Trend: 1 = bullish (price above SMA200), -1 = bearish (price below SMA200), 0 = neutral/no trade
    trend_1w = np.where(close_1w > sma_200_1w, 1, np.where(close_1w < sma_200_1w, -1, 0))
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Get daily data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # Daily Donchian(20) channels
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max()
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min()
    # Breakout signals
    donch_breakout_up = (high_1d > donch_high_20.shift(1))  # New 20-day high
    donch_breakout_down = (low_1d < donch_low_20.shift(1))   # New 20-day low
    donch_breakout_up_aligned = align_htf_to_ltf(prices, df_1d, donch_breakout_up)
    donch_breakout_down_aligned = align_htf_to_ltf(prices, df_1d, donch_breakout_down)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not session_mask[i] or \
           np.isnan(trend_1w_aligned[i]) or \
           np.isnan(donch_breakout_up_aligned[i]) or \
           np.isnan(donch_breakout_down_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Get current trend and breakout signals
        trend = trend_1w_aligned[i]
        breakout_up = donch_breakout_up_aligned[i]
        breakout_down = donch_breakout_down_aligned[i]
        
        # Volume confirmation on 12h
        if i >= 20:
            vol_ma_20 = pd.Series(volume[:i+1]).rolling(window=20, min_periods=20).mean().iloc[-1]
            volume_expansion = volume[i] > (vol_ma_20 * 1.5)
        else:
            volume_expansion = False
        
        # Trading logic: only trade in direction of weekly trend
        if trend == 1:  # Bullish trend - look for long breakouts
            if breakout_up and volume_expansion:
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
            elif position == 1:
                # Exit long if bearish breakout occurs
                if breakout_down:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:
                signals[i] = 0.0
        elif trend == -1:  # Bearish trend - look for short breakouts
            if breakout_down and volume_expansion:
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
            elif position == -1:
                # Exit short if bullish breakout occurs
                if breakout_up:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:
                signals[i] = 0.0
        else:
            # No clear trend - stay flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_1w_Triple_Timeframe_Donchian"
timeframe = "12h"
leverage = 1.0