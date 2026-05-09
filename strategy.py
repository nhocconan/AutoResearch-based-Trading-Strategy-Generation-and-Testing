#!/usr/bin/env python3
"""
6h_Weekly_Open_Gap_Fill
Trades gaps between weekly open and close using 6h candles.
Hypothesis: Weekly gaps (between Friday close and Monday open) tend to fill during the week.
We enter on the first 6h candle after weekly open if price gaps beyond the weekly open,
and target a return to the weekly close (gap fill). Uses 1d trend filter to avoid counter-trend trades.
Works in both bull and bear markets as gap filling is a mean-reverting behavior.
Target: ~20-50 trades over 4 years (5-12/year) with size 0.25.
"""

name = "6h_Weekly_Open_Gap_Fill"
timeframe = "6h"
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
    open_ = prices['open'].values
    volume = prices['volume'].values
    
    # Get weekly data for gap detection
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly open and close
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    
    # Align weekly levels to 6h timeframe (with 1-bar delay for weekly close)
    weekly_open_aligned = align_htf_to_ltf(prices, df_1w, weekly_open)
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close, additional_delay_bars=1)
    
    # Get daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA20 for trend filter
    ema_20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    trend_up = close > ema_20_1d_aligned
    trend_down = close < ema_20_1d_aligned
    
    # Volume filter: avoid low-volume periods
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24*6h = 6 days
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup period
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_open_aligned[i]) or np.isnan(weekly_close_aligned[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Gap up: weekly open > previous weekly close -> look for fill (short)
            gap_up = weekly_open_aligned[i] > weekly_close_aligned[i]
            # Gap down: weekly open < previous weekly close -> look for fill (long)
            gap_down = weekly_open_aligned[i] < weekly_close_aligned[i]
            
            # Enter short on gap up if price is above weekly open and in downtrend
            if gap_up and open_[i] > weekly_open_aligned[i] and trend_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            # Enter long on gap down if price is below weekly open and in uptrend
            elif gap_down and open_[i] < weekly_open_aligned[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Long: target gap fill (return to weekly close)
            if close[i] >= weekly_close_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short: target gap fill (return to weekly close)
            if close[i] <= weekly_close_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals