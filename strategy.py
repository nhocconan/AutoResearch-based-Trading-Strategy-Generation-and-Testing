#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_Filter_V3
Hypothesis: Trade 4-hour Donchian channel breakouts with volume confirmation and daily EMA50 trend filter.
Long when price breaks above 20-period Donchian high with volume spike and daily uptrend (price > EMA50);
short when breaks below 20-period Donchian low with volume spike and daily downtrend (price < EMA50).
Uses proven Donchian breakout logic from top performers, with volume to reduce false breakouts and trend filter to avoid counter-trend trades.
Target: 80-160 total trades over 4 years (20-40/year) with position size 0.25 to balance opportunity and risk.
Works in bull/bear: daily trend filter ensures we only trade with the higher timeframe trend, reducing whipsaw in ranging markets.
"""

name = "4h_Donchian_Breakout_Volume_Trend_Filter_V3"
timeframe = "4h"
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
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA50 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema50_1d = ema(close_1d, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 20-period Donchian channels
    def donchian_channel(high, low, period):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donchian_high, donchian_low = donchian_channel(high, low, 20)
    
    # Calculate volume spike (volume > 1.5x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume spike AND daily uptrend
            if close[i] > donchian_high[i] and volume_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume spike AND daily downtrend
            elif close[i] < donchian_low[i] and volume_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR daily trend turns down
            if close[i] < donchian_low[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR daily trend turns up
            if close[i] > donchian_high[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals