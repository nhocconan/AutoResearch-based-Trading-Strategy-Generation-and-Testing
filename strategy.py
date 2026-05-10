#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_Filter
Hypothesis: Uses Donchian channel breakout (20-period) on 4h timeframe with volume confirmation
and 1-day EMA50 trend filter to capture strong momentum moves in both bull and bear markets.
Designed for low trade frequency (20-30/year) to minimize fee drift while capturing trends.
"""

name = "4h_Donchian_Breakout_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donch_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    # Use pandas rolling for vectorized min/max with min_periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=donch_period, min_periods=donch_period).max().values
    lower = low_series.rolling(window=donch_period, min_periods=donch_period).min().values
    
    # 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation (20-period average)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = donch_period + 5  # Ensure indicators are warm
    
    for i in range(start_idx, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume, above 1-day EMA50
            if close[i] > upper[i] and volume_confirm and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band with volume, below 1-day EMA50
            elif close[i] < lower[i] and volume_confirm and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below lower Donchian band or breaks below 1-day EMA50
            if close[i] < lower[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above upper Donchian band or breaks above 1-day EMA50
            if close[i] > upper[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals