#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 12h pivot direction filter and volume confirmation
    # In 2025-2026 bear/range market, breakouts with institutional volume and higher timeframe
    # directional bias (from 12h pivot levels) reduce false signals. Works in both bull/bear
    # by requiring volume confirmation and trend alignment. Target: 15-30 trades/year.
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 6h data for Donchian channels (primary timeframe structure)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate 6h Donchian Channels (20-period)
    donchian_period = 20
    upper_donchian = pd.Series(high_6h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_donchian = pd.Series(low_6h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Align Donchian Channels to 6h timeframe
    upper_donchian_aligned = align_htf_to_ltf(prices, df_6h, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_6h, lower_donchian)
    
    # Load 12h data for pivot-based trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Pivot Points (standard formula)
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    r1_12h = 2 * pivot_12h - low_12h
    s1_12h = 2 * pivot_12h - high_12h
    r2_12h = pivot_12h + (high_12h - low_12h)
    s2_12h = pivot_12h - (high_12h - low_12h)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(upper_donchian_aligned[i]) or np.isnan(lower_donchian_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(vol_ma20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above upper Donchian with volume + price above 12h R1 (bullish bias)
            if close[i] > upper_donchian_aligned[i] and vol_spike[i] and close[i] > r1_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below lower Donchian with volume + price below 12h S1 (bearish bias)
            elif close[i] < lower_donchian_aligned[i] and vol_spike[i] and close[i] < s1_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level or crosses 12h pivot (trend change)
            if position == 1:
                if close[i] < lower_donchian_aligned[i] or close[i] < pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_donchian_aligned[i] or close[i] > pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian_Breakout_12hPivot_Volume_Session_v1"
timeframe = "6h"
leverage = 1.0