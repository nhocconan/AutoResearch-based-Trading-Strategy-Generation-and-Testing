#!/usr/bin/env python3
# 1d_1w_Donchian_Breakout_Trend_Weekly
# Hypothesis: Weekly Donchian breakouts with daily trend alignment capture major trends in both bull and bear markets.
# Uses weekly price channels to avoid whipsaw, daily EMA for trend filter, and volume confirmation for institutional participation.
# Target: 10-25 trades/year to minimize fee drag while capturing major moves.

name = "1d_1w_Donchian_Breakout_Trend_Weekly"
timeframe = "1d"
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
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly upper channel: highest high of last 20 weeks
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Weekly lower channel: lowest low of last 20 weeks
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly channels to daily timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Daily trend filter: EMA(50)
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).values
    
    # Volume filter: volume > 1.5x 20-day EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > (vol_ema20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or
            np.isnan(ema_50[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly upper channel + above daily EMA50 + volume
            if close[i] > upper_20_aligned[i] and close[i] > ema_50[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly lower channel + below daily EMA50 + volume
            elif close[i] < lower_20_aligned[i] and close[i] < ema_50[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly lower channel or below EMA50
            if close[i] < lower_20_aligned[i] or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly upper channel or above EMA50
            if close[i] > upper_20_aligned[i] or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals