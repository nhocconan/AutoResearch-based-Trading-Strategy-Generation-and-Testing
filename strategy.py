#!/usr/bin/env python3
# 6h_12h_1d_Donchian_Breakout_Trend_Volume
# Hypothesis: 6h Donchian(20) breakout with 12h trend filter (EMA50) and volume confirmation.
# Uses 12h for trend direction, 6s for entry timing. Works in bull/bear by requiring trend alignment,
# avoiding counter-trend traps. Targets 50-150 total trades over 4 years via strict multi-condition entry.

name = "6h_12h_1d_Donchian_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data for trend filter and Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    high_20 = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe (wait for 12h bar to close)
    dc_high_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    dc_low_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume average (4-period for 6h = 1 day)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for Donchian (20 periods) + EMA + vol MA
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(dc_high_aligned[i]) or
            np.isnan(dc_low_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend: 12h close > EMA50
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, df_12h['close'].values)
        uptrend = close_12h_aligned[i] > ema_50_12h_aligned[i]
        downtrend = close_12h_aligned[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation (1.5x average)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Breakout above Donchian high in uptrend with volume
            if close[i] > dc_high_aligned[i] and uptrend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below Donchian low in downtrend with volume
            elif close[i] < dc_low_aligned[i] and downtrend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: close below Donchian high or trend fails
                if close[i] < dc_high_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: close above Donchian low or trend fails
                if close[i] > dc_low_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals