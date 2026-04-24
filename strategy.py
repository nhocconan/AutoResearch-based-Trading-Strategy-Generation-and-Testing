#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for weekly pivot calculation (R1/S1, R2/S2, R3/S3, R4/S4 from prior week).
- Entry: Long when price breaks above Donchian(20) high AND weekly pivot trend is bullish (price > weekly PP) AND volume > 1.5x 20-period MA.
         Short when price breaks below Donchian(20) low AND weekly pivot trend is bearish (price < weekly PP) AND volume > 1.5x 20-period MA.
- Exit: Opposite Donchian breakout OR price crosses weekly pivot point (PP) in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Weekly pivot provides structural support/resistance from higher timeframe.
- Donchian breakout captures momentum with volume confirmation to avoid false breakouts.
- Works in bull markets (buy breakouts above PP) and bear markets (sell breakdowns below PP).
- Estimated trades: ~100 total over 4 years (~25/year) based on breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Calculate 1d weekly pivot points from prior week
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Need at least 5 days for prior week (Monday-Friday)
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Get prior week's OHLC (excluding current incomplete week)
    # We use the week that ended 1-5 days ago to avoid look-ahead
    prior_week_high = np.full(len(df_1d), np.nan)
    prior_week_low = np.full(len(df_1d), np.nan)
    prior_week_close = np.full(len(df_1d), np.nan)
    
    for i in range(5, len(df_1d)):  # Start from index 5 to have 5-day prior week
        # Prior week: 5 days ending yesterday (i-1 to i-5)
        week_high = np.max(df_1d['high'].values[i-5:i])
        week_low = np.min(df_1d['low'].values[i-5:i])
        week_close = df_1d['close'].values[i-1]  # Yesterday's close
        
        prior_week_high[i] = week_high
        prior_week_low[i] = week_low
        prior_week_close[i] = week_close
    
    # Calculate weekly pivot points (using prior week's data)
    pp = (prior_week_high + prior_week_low + prior_week_close) / 3
    r1 = 2 * pp - prior_week_low
    s1 = 2 * pp - prior_week_high
    r2 = pp + (prior_week_high - prior_week_low)
    s2 = pp - (prior_week_high - prior_week_low)
    r3 = pp + 2 * (prior_week_high - prior_week_low)
    s3 = pp - 2 * (prior_week_high - prior_week_low)
    r4 = pp + 3 * (prior_week_high - prior_week_low)
    s4 = pp - 3 * (prior_week_high - prior_week_low)
    
    # Align weekly pivot levels to 6h timeframe (completed weekly bar only)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp, additional_delay_bars=0)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1, additional_delay_bars=0)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1, additional_delay_bars=0)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2, additional_delay_bars=0)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2, additional_delay_bars=0)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3, additional_delay_bars=0)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3, additional_delay_bars=0)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4, additional_delay_bars=0)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4, additional_delay_bars=0)
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):  # 20-period MA needs 20 bars
        vol_ma[i] = np.mean(volume[i-19:i+1])
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback - 1, 24)  # Need Donchian + weekly pivot alignment
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(pp_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses weekly PP in opposite direction
        if position != 0:
            # Exit long: price breaks below Donchian low OR price falls below weekly PP
            if position == 1:
                if curr_low < lowest_low[i] or curr_close < pp_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian high OR price rises above weekly PP
            elif position == -1:
                if curr_high > highest_high[i] or curr_close > pp_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout + volume confirmation + weekly pivot alignment
        if position == 0:
            # Long: price breaks above Donchian high AND volume filter AND bullish weekly pivot (price > PP)
            if curr_high > highest_high[i] and volume_filter[i] and curr_close > pp_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND volume filter AND bearish weekly pivot (price < PP)
            elif curr_low < lowest_low[i] and volume_filter[i] and curr_close < pp_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0