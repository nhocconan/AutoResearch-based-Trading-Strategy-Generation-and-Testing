#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction filter and volume confirmation.
Long when price breaks above 6h Donchian upper band AND 1d weekly pivot shows bullish bias AND volume > 1.5x 20-period average.
Short when price breaks below 6h Donchian lower band AND 1d weekly pivot shows bearish bias AND volume > 1.5x 20-period average.
Exit when price touches the opposite Donchian band or weekly pivot bias reverses.
Uses 1d HTF for weekly pivot bias (avoids whipsaws in ranging markets). Target: 50-150 total trades over 4 years (12-37/year).
Donchian channels provide structure; weekly pivot filter ensures we trade with the dominant higher timeframe bias.
"""

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
    
    # Calculate 6h Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d weekly pivot points (using prior week's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least 5 days for prior week
        return np.zeros(n)
    
    # Get prior week's OHLC (Monday to Friday of previous week)
    # We'll approximate using the last 5 daily bars excluding current
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot using prior 5 days (simplified: use last completed week)
    # For each point, use high/low/close from 5 days ago to 1 day ago
    if len(high_1d) >= 6:
        # Weekly high = max of prior 5 days high
        weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values
        # Weekly low = min of prior 5 days low
        weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
        # Weekly close = close of prior 1 day (yesterday)
        weekly_close = pd.Series(close_1d).shift(1).values
    else:
        weekly_high = np.full(len(high_1d), np.nan)
        weekly_low = np.full(len(high_1d), np.nan)
        weekly_close = np.full(len(close_1d), np.nan)
    
    # Calculate weekly pivot point and support/resistance levels
    # PP = (H + L + C) / 3
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
    # R1 = (2 * PP) - L
    weekly_r1 = 2 * weekly_pp - weekly_low
    # S1 = (2 * PP) - H
    weekly_s1 = 2 * weekly_pp - weekly_high
    # R2 = PP + (H - L)
    weekly_r2 = weekly_pp + (weekly_high - weekly_low)
    # S2 = PP - (H - L)
    weekly_s2 = weekly_pp - (weekly_high - weekly_low)
    # R3 = H + 2*(PP - L)
    weekly_r3 = weekly_high + 2 * (weekly_pp - weekly_low)
    # S3 = L - 2*(H - PP)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pp)
    
    # Align weekly pivot levels to 6h timeframe (use prior week's levels)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1d, weekly_pp, additional_delay_bars=1)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3, additional_delay_bars=1)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3, additional_delay_bars=1)
    
    # Determine weekly pivot bias: bullish if price > PP, bearish if price < PP
    # We'll use the aligned weekly PP to determine bias
    weekly_bias_bullish = None  # Will compute inside loop
    weekly_bias_bearish = None
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Donchian (20), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(weekly_pp_aligned[i]) or np.isnan(weekly_r3_aligned[i]) or 
            np.isnan(weekly_s3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        pp = weekly_pp_aligned[i]
        r3 = weekly_r3_aligned[i]
        s3 = weekly_s3_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Determine weekly pivot bias at this point
        weekly_bias_bullish = price > pp
        weekly_bias_bearish = price < pp
        
        if position == 0:
            # Long: Break above Donchian upper AND weekly pivot bullish AND volume spike
            if price > upper and weekly_bias_bullish and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower AND weekly pivot bearish AND volume spike
            elif price < lower and weekly_bias_bearish and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Donchian lower OR weekly pivot turns bearish
                if price < lower or not weekly_bias_bullish:
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Donchian upper OR weekly pivot turns bullish
                if price > upper or not weekly_bias_bearish:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_Breakout_1dWeeklyPivot_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0