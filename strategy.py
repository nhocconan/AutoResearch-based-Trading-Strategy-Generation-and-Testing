#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_Volume
Hypothesis: Use 12-hour Donchian(20) breakout with daily trend filter (price > EMA50 for long, < EMA50 for short) and volume confirmation (>1.5x 20-period EMA volume). Designed for low trade frequency (<30/year) to minimize fee drag on 12h timeframe. Works in bull/bear by only taking breakouts in the direction of the daily trend.
"""

name = "12h_Donchian20_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # === Get Daily Data for Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA50 trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 12h Donchian(20) Channel ===
    # Calculate Donchian channels on 12h data directly (no HTF needed for this)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # === Volume Spike Filter (1.5x 20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers Donchian and EMA calculation)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with uptrend (close > EMA50) and volume spike
            if (high[i] > donchian_high[i] and 
                close[i] > ema50_1d_aligned[i] and volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: Price breaks below Donchian low with downtrend (close < EMA50) and volume spike
            elif (low[i] < donchian_low[i] and 
                  close[i] < ema50_1d_aligned[i] and volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Price crosses back through the opposite Donchian level
            if position == 1:
                if low[i] < donchian_low[i]:  # Exit long if price breaks below Donchian low
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if high[i] > donchian_high[i]:  # Exit short if price breaks above Donchian high
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals