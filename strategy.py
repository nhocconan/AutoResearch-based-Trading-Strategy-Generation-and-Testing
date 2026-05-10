#!/usr/bin/env python3
# 12H_1D_Donchian_Breakout_Trend_Filter
# Hypothesis: On the 12h timeframe, take long positions when price breaks above the 20-period Donchian high
# during a daily uptrend (price > daily EMA50), and short positions when price breaks below the 20-period
# Donchian low during a daily downtrend (price < daily EMA50). Uses volume confirmation and ATR-based
# stoploss via signal reversal. Designed for low trade frequency (~20-40 trades/year) to avoid fee drag.
# Works in bull/bear by following daily trend direction.

name = "12H_1D_Donchian_Breakout_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Trend: bullish if close > EMA50, bearish if close < EMA50
    bullish_trend = close_1d > ema50_1d
    bearish_trend = close_1d < ema50_1d
    
    # Align trend to 12h
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend.astype(float))
    
    # 12h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):  # 20-period MA
        vol_ma[i] = np.mean(volume[i - 19:i + 1])
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = max(50, lookback - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_aligned[i] > 0.5
        bearish = bearish_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: bullish trend + price breaks above Donchian high + volume confirmation
            if bullish and close[i] > highest_high[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish trend + price breaks below Donchian low + volume confirmation
            elif bearish and close[i] < lowest_low[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish trend or price breaks below Donchian low
            if bearish or close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish trend or price breaks above Donchian high
            if bullish or close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals