#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Filter_v2
Hypothesis: On 4h timeframe, enter long when price breaks above Donchian(20) high with volume confirmation; enter short when price breaks below Donchian(20) low with volume confirmation. Uses 1d close trend filter (close > open) to avoid counter-trend trades. Designed for low trade frequency (20-40/year) to minimize fee drift while capturing strong trending moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Donchian Channel (20-period) on 4h ===
    donchian_window = 20
    high_roll = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max()
    low_roll = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    
    # === 1d close trend filter (bullish daily candle) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    daily_bullish = close_1d > open_1d  # Bullish daily candle
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    
    # === 1d volume average for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = max(donchian_window, 20)
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(daily_bullish_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current daily bar's volume for confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filter: current volume > 1.5x daily average volume
        vol_filter = vol_1d_current > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Daily trend filter: only take longs on bullish days, shorts on bearish days
        daily_bull = daily_bullish_aligned[i]
        daily_bear = not daily_bull
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above Donchian high + volume filter + bullish daily candle
            if close[i] > donchian_high[i] and vol_filter and daily_bull:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below Donchian low + volume filter + bearish daily candle
            elif close[i] < donchian_low[i] and vol_filter and daily_bear:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit when price closes below Donchian low (reversal signal)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price closes above Donchian high (reversal signal)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Filter_v2"
timeframe = "4h"
leverage = 1.0