#!/usr/bin/env python3
# 4h_Donchian_20_Breakout_12hEMA50_Trend_Volume
# Hypothesis: 4-hour Donchian(20) breakouts aligned with 12-hour EMA50 trend and volume confirmation
# work in both bull and bear markets by capturing institutional breakouts while avoiding false signals.
# Target: 20-50 trades/year to minimize fee drag on 4h timeframe.

name = "4h_Donchian_20_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = close_12h > ema50_12h
    trend_12h_down = close_12h < ema50_12h
    
    # Align 12h trend to 4h
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down.astype(float))
    
    # Donchian channel (20-period high/low)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 70  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume confirmation and 12h uptrend
            if (high[i] > donchian_high[i] and
                trend_12h_up_aligned[i] > 0.5 and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume confirmation and 12h downtrend
            elif (low[i] < donchian_low[i] and
                  trend_12h_down_aligned[i] > 0.5 and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below Donchian low or 12h trend turns down
            if (low[i] < donchian_low[i] or
                trend_12h_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above Donchian high or 12h trend turns up
            if (high[i] > donchian_high[i] or
                trend_12h_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals