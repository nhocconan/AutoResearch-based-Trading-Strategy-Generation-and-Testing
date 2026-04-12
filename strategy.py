#!/usr/bin/env python3
"""
4h_1D_Heikin_Ashi_Trend_Strategy
Hypothesis: Heikin Ashi (HA) candles on daily timeframe filter noise and reveal true trend direction.
Combined with 4h price action: long when HA close > HA open (bullish candle) and price breaks above 4h high of previous 3 candles,
short when HA close < HA open (bearish candle) and price breaks below 4h low of previous 3 candles.
Uses volume confirmation to avoid false breakouts. Designed for low trade frequency (~20-30/year) to minimize fee drag.
Works in both bull (rides uptrends) and bear (captures downtrends) markets by following HA trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1D_Heikin_Ashi_Trend_Strategy"
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
    
    # === DAILY DATA FOR HEIKIN ASHI ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Heikin Ashi calculations
    ha_close = (open_1d + high_1d + low_1d + close_1d) / 4
    ha_open = np.zeros_like(ha_close)
    ha_open[0] = (open_1d[0] + close_1d[0]) / 2
    for i in range(1, len(ha_open)):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    
    # HA trend: bullish when ha_close > ha_open, bearish when ha_close < ha_open
    ha_bullish = ha_close > ha_open
    ha_bearish = ha_close < ha_open
    
    # Align HA trend to 4h timeframe (wait for daily bar to close)
    ha_bullish_aligned = align_htf_to_ltf(prices, df_1d, ha_bullish.astype(float))
    ha_bearish_aligned = align_htf_to_ltf(prices, df_1d, ha_bearish.astype(float))
    
    # === 4H INDICATORS ===
    # High and low of previous 3 candles for breakout levels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    # Rolling max of high for last 3 periods (excluding current)
    high_3bar_max = high_series.rolling(window=3, min_periods=1).max().shift(1).values
    low_3bar_min = low_series.rolling(window=3, min_periods=1).min().shift(1).values
    
    # Volume confirmation (4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(ha_bullish_aligned[i]) or np.isnan(ha_bearish_aligned[i]) or 
            np.isnan(high_3bar_max[i]) or np.isnan(low_3bar_min[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long setup: Daily HA bullish + price breaks above 3-bar high + volume confirmation
        long_setup = (ha_bullish_aligned[i] > 0.5) and (close[i] > high_3bar_max[i]) and (vol_ratio[i] > 1.3)
        
        # Short setup: Daily HA bearish + price breaks below 3-bar low + volume confirmation
        short_setup = (ha_bearish_aligned[i] > 0.5) and (close[i] < low_3bar_min[i]) and (vol_ratio[i] > 1.3)
        
        # Exit when daily HA trend changes
        exit_long = ha_bearish_aligned[i] > 0.5  # HA turned bearish
        exit_short = ha_bullish_aligned[i] > 0.5  # HA turned bullish
        
        # Execute trades
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals