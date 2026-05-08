#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 1d trend filter.
# Uses 4h Donchian channels for breakout detection, 1d EMA50 for trend direction,
# and volume spike (>2x average) for confirmation.
# Long when price breaks above upper Donchian band + 1d trend up + volume spike.
# Short when price breaks below lower Donchian band + 1d trend down + volume spike.
# Exit on opposite Donchian band touch or trend reversal.
# Designed to work in bull markets (breakouts) and bear markets (breakdowns).
# Target: 20-50 trades per year to minimize fee drag.

name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data (same timeframe, but needed for calculations)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 4h Donchian channels (20-period)
    donchian_period = 20
    upper = np.full_like(high_4h, np.nan)
    lower = np.full_like(low_4h, np.nan)
    
    for i in range(donchian_period - 1, len(high_4h)):
        upper[i] = np.max(high_4h[i - donchian_period + 1:i + 1])
        lower[i] = np.min(low_4h[i - donchian_period + 1:i + 1])
    
    # 1d EMA(50) for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = ema_50_1d > np.roll(ema_50_1d, 1)
    trend_1d_up = np.where(np.isnan(trend_1d_up), False, trend_1d_up)
    
    # Align 1d trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, donchian_period - 1)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian + 1d trend up + volume spike
            if (close[i] > upper[i] and trend_1d_up_aligned[i] and vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + 1d trend down + volume spike
            elif (close[i] < lower[i] and not trend_1d_up_aligned[i] and vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price touches lower Donchian or 1d trend turns down
            if (close[i] < lower[i] or not trend_1d_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches upper Donchian or 1d trend turns up
            if (close[i] > upper[i] or trend_1d_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals