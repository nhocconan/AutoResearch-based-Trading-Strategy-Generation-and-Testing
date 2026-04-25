#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dEMA34_Trend_VolumeConfirm_v1
Hypothesis: Trade Donchian(20) breakouts on 4h with 1d EMA34 trend filter and volume confirmation. In bullish 1d trend, buy breakouts above upper channel; in bearish 1d trend, sell breakdowns below lower channel. Volume confirmation (1.5x 20-bar avg) filters false breakouts. Designed for 4h timeframe with ~25-40 trades/year to minimize fee drag while capturing strong directional moves. Works in both bull (trend continuation) and bear (trend continuation short) markets.
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
    
    # Get 1d data for HTF trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian(20) channels on 4h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 1.5x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian(20) and volume MA(20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(high_roll[i]) or
            np.isnan(low_roll[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for Donchian breakouts with volume confirmation
            long_breakout = (high[i] > high_roll[i-1]) and volume_spike[i]  # break above prev period high
            short_breakout = (low[i] < low_roll[i-1]) and volume_spike[i]   # break below prev period low
            
            # Only trade in direction of 1d trend
            if long_breakout and htf_1d_bullish:
                signals[i] = 0.25
                position = 1
            elif short_breakout and htf_1d_bearish:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price returns to Donchian midpoint or trend reverses
            midpoint = (high_roll[i] + low_roll[i]) / 2.0
            exit_signal = (close[i] < midpoint) or (not htf_1d_bullish)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price returns to Donchian midpoint or trend reverses
            midpoint = (high_roll[i] + low_roll[i]) / 2.0
            exit_signal = (close[i] > midpoint) or htf_1d_bullish
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0