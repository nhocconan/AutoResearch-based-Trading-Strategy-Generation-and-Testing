#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_Filter_v1
Hypothesis: On 4h timeframe, enter long when price breaks above Donchian(20) high with volume confirmation and daily trend alignment; enter short when price breaks below Donchian(20) low with volume confirmation and daily trend alignment. Uses 1d EMA50 as trend filter to avoid counter-trend trades. Designed for low trade frequency (20-40/year) to minimize fee drag while capturing strong trending moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # === 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d volume average for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = max(donchian_window, 50, 20)
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current daily bar's volume for confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filter: current volume > 1.8x daily average volume
        vol_filter = vol_1d_current > 1.8 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: price above/below EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above Donchian high + volume filter + price above EMA50
            if close[i] > donchian_high[i] and vol_filter and price_above_ema:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below Donchian low + volume filter + price below EMA50
            elif close[i] < donchian_low[i] and vol_filter and price_below_ema:
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

name = "4h_Donchian_Breakout_Volume_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0