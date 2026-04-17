#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_Volume_Trend_1d
Hypothesis: On 12h timeframe, enter long when price breaks above 20-bar Donchian high with volume confirmation and daily close above daily EMA50; enter short when price breaks below 20-bar Donchian low with volume confirmation and daily close below daily EMA50. Uses daily EMA50 as trend filter to align with higher timeframe trend. Designed for low trade frequency (12-37/year) to minimize fee decay while capturing strong trending moves in both bull and bear markets.
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
    
    # === 12h Donchian Channels (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d EMA50 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 1d volume average for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 20  # For Donchian and volume average
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current daily bar's volume for confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filter: current volume > 1.5x daily average volume
        vol_filter = vol_1d_current > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Daily trend filters
        daily_uptrend = close[i] > ema50_1d_aligned[i]
        daily_downtrend = close[i] < ema50_1d_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: break above Donchian high + volume filter + daily uptrend
            if close[i] > donchian_high[i] and vol_filter and daily_uptrend:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below Donchian low + volume filter + daily downtrend
            elif close[i] < donchian_low[i] and vol_filter and daily_downtrend:
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

name = "12h_Donchian20_Breakout_Volume_Trend_1d"
timeframe = "12h"
leverage = 1.0