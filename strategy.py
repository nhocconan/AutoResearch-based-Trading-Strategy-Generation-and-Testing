#!/usr/bin/env python3
# 6h_Engulfing_1dTrend_Volume
# Hypothesis: Uses daily bullish/bearish engulfing candles to signal trend direction, combined with 6-hour price action and volume confirmation.
# Engulfing candles on the daily chart indicate strong institutional sentiment. We only take trades in the direction of the daily engulfing candle,
# entering on 6-hour breakouts of the engulfing candle's body with volume confirmation. This filters noise and aligns with higher timeframe momentum.
# Works in both bull and bear markets by following the daily trend. Target: 15-30 trades/year per symbol to minimize fee drag.

timeframe = "6h"
name = "6h_Engulfing_1dTrend_Volume"
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
    
    # Get daily data for engulfing candle detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    
    # Bullish engulfing: current day closes above prior day's open AND opens below prior day's close
    bullish_engulf = (close_1d > open_1d) & (open_1d < close_1d) & (close_1d > open_1d) & (open_1d < close_1d)
    bullish_engulf = (close_1d > open_1d.shift(1)) & (open_1d < close_1d.shift(1))
    # Bearish engulfing: current day closes below prior day's open AND opens above prior day's close
    bearish_engulf = (close_1d < open_1d.shift(1)) & (open_1d > close_1d.shift(1))
    
    # Align engulfing signals to 6h timeframe (use the engulfing candle's signal for the entire day)
    bullish_engulf_aligned = align_htf_to_ltf(prices, df_1d, bullish_engulf.astype(float))
    bearish_engulf_aligned = align_htf_to_ltf(prices, df_1d, bearish_engulf.astype(float))
    
    # 6-hour moving average for dynamic support/resistance (20-period = ~5 days)
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: 1.5x average volume (6-period = 1 day on 6h chart)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 6)  # Ensure we have MA and volume data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(bullish_engulf_aligned[i]) or np.isnan(bearish_engulf_aligned[i]) or 
            np.isnan(ma_20[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: daily bullish engulfing + price above 6h MA + volume confirmation
            if bullish_engulf_aligned[i] > 0.5 and close[i] > ma_20[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: daily bearish engulfing + price below 6h MA + volume confirmation
            elif bearish_engulf_aligned[i] > 0.5 and close[i] < ma_20[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below 6h MA (trend reversal)
            if close[i] < ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above 6h MA (trend reversal)
            if close[i] > ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals