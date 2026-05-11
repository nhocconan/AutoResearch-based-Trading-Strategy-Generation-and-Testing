#!/usr/bin/env python3
# 4h_Donchian20_1dTrend_Volume
# Hypothesis: Combine 4h Donchian(20) breakout with 1d EMA50 trend and volume confirmation.
# Long when price breaks above Donchian(20) high, 1d EMA50 rising, and volume > 20-period average.
# Short when price breaks below Donchian(20) low, 1d EMA50 falling, and volume > 20-period average.
# Uses volume to avoid false breakouts and follows higher timeframe trend for bias.
# Works in bull markets (riding uptrends) and bear markets (riding downtrends) by following the higher timeframe trend.
# Target: 20-50 trades/year per symbol to avoid fee drag.

name = "4h_Donchian20_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d EMA50 for trend direction ---
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_slope = ema_50_1d - np.roll(ema_50_1d, 1)
    ema_50_1d_slope[0] = 0
    ema_50_1d_slope = pd.Series(ema_50_1d_slope).ewm(span=3, adjust=False, min_periods=1).mean().values  # smooth slope
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_50_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_slope)
    
    # --- 4h Donchian(20) channels ---
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Donchian(20) and EMA50 slope
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema_50_1d_slope_aligned[i]) or
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from 1d EMA50 slope
        uptrend = ema_50_1d_slope_aligned[i] > 0
        downtrend = ema_50_1d_slope_aligned[i] < 0
        
        if position == 0:
            if uptrend and vol_surge[i]:
                # Long: 1d uptrend + volume surge + price breaks above Donchian high
                if close[i] > highest_20[i]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_surge[i]:
                # Short: 1d downtrend + volume surge + price breaks below Donchian low
                if close[i] < lowest_20[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: 1d trend turns down OR price breaks below Donchian low
                if downtrend or close[i] < lowest_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: 1d trend turns up OR price breaks above Donchian high
                if uptrend or close[i] > highest_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals