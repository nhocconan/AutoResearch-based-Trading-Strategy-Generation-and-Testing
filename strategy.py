#!/usr/bin/env python3
name = "6h_WeeklyPivot_DonchianBreakout_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data once for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate upper/lower bands using rolling window
    high_max_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align to 6h timeframe
    donchian_upper = align_htf_to_ltf(prices, df_12h, high_max_20)
    donchian_lower = align_htf_to_ltf(prices, df_12h, low_min_20)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection (6-period average)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition
        vol_condition = volume[i] > vol_ma_6[i] * 2.0
        
        if position == 0:
            # Long: Donchian breakout above upper band in uptrend
            if close[i] > donchian_upper[i] and vol_condition and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout below lower band in downtrend
            elif close[i] < donchian_lower[i] and vol_condition and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Donchian lower band or trend reversal
            if close[i] < donchian_lower[i] or ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Donchian upper band or trend reversal
            if close[i] > donchian_upper[i] or ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Donchian breakout with 12h channels, 1d trend filter, and volume confirmation
# - Uses 12h Donchian channels (20-period) for breakout signals
# - 1d EMA50 trend filter ensures trades align with higher timeframe trend
# - Volume confirmation (2x 6-period average) reduces false breakouts
# - Works in both bull (breakouts above upper band in uptrend) and bear (breakdowns below lower band in downtrend)
# - Position size 0.25 targets ~50-150 total trades over 4 years (12-37/year) to avoid fee drag
# - Donchian breakouts provide clear entry/exit levels with built-in trend following
# - Multi-timeframe approach (12h channels + 1d trend) reduces whipsaws vs single timeframe
# - Novel combination: 12h Donchian + 1d trend + volume filter not recently tried in 6h timeframe
# - Aims for 60-120 total trades over 4 years (15-30/year) to stay within limits
# - Effective in both trending and ranging markets due to trend filter and volatility breakout logic