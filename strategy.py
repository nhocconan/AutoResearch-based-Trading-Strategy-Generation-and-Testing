#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d volume spike and weekly pivot trend filter.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for volume confirmation, 1w for pivot-based trend direction.
- Camarilla Pivots: calculates key levels from prior 1d OHLC.
- Entry: Long when price breaks above R3 AND volume > 1.8 * 20-period average volume (1d) AND weekly pivot shows uptrend (price > weekly pivot).
         Short when price breaks below S3 AND volume > 1.8 * 20-period average volume (1d) AND weekly pivot shows downtrend (price < weekly pivot).
- Exit: Opposite Camarilla breakout (R4/S4) or reversal signal.
- Signal size: 0.25 discrete to minimize fee drag.
- Weekly pivot trend filter avoids counter-trend trades in strong weekly regimes.
- Volume confirmation ensures breakout legitimacy.
- Designed to work in both bull and bear markets by aligning with weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC."""
    pivot = (high + low + close) / 3.0
    range_hl = high - low
    r3 = pivot + range_hl * 1.1 / 4.0
    s3 = pivot - range_hl * 1.1 / 4.0
    r4 = pivot + range_hl * 1.1 / 2.0
    s4 = pivot - range_hl * 1.1 / 2.0
    return r3, s3, r4, s4, pivot

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels (using prior 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for prior bar
        return np.zeros(n)
    
    # Use prior 1d bar for Camarilla calculation (avoid look-ahead)
    prior_high = df_1d['high'].values[:-1]  # exclude current forming bar
    prior_low = df_1d['low'].values[:-1]
    prior_close = df_1d['close'].values[:-1]
    
    if len(prior_close) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each prior 1d bar
    r3_1d = np.full_like(prior_close, np.nan)
    s3_1d = np.full_like(prior_close, np.nan)
    r4_1d = np.full_like(prior_close, np.nan)
    s4_1d = np.full_like(prior_close, np.nan)
    pivot_1d = np.full_like(prior_close, np.nan)
    
    for i in range(len(prior_close)):
        r3, s3, r4, s4, pivot = camarilla_pivot(prior_high[i], prior_low[i], prior_close[i])
        r3_1d[i] = r3
        s3_1d[i] = s3
        r4_1d[i] = r4
        s4_1d[i] = s4
        pivot_1d[i] = pivot
    
    # Align 1d levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d.iloc[:-1], r3_1d) if len(df_1d) > 1 else np.full(n, np.nan)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d.iloc[:-1], s3_1d) if len(df_1d) > 1 else np.full(n, np.nan)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d.iloc[:-1], r4_1d) if len(df_1d) > 1 else np.full(n, np.nan)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d.iloc[:-1], s4_1d) if len(df_1d) > 1 else np.full(n, np.nan)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d.iloc[:-1], pivot_1d) if len(df_1d) > 1 else np.full(n, np.nan)
    
    # Calculate 1d volume average for confirmation
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1w pivot for trend filter (using prior 1w bar)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Use prior 1w bar for weekly pivot
    prior_week_high = df_1w['high'].values[:-1]
    prior_week_low = df_1w['low'].values[:-1]
    prior_week_close = df_1w['close'].values[:-1]
    
    if len(prior_week_close) < 1:
        weekly_pivot_aligned = np.full(n, np.nan)
    else:
        weekly_pivot_1w = np.full_like(prior_week_close, np.nan)
        for i in range(len(prior_week_close)):
            weekly_pivot = (prior_week_high[i] + prior_week_low[i] + prior_week_close[i]) / 3.0
            weekly_pivot_1w[i] = weekly_pivot
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w.iloc[:-1], weekly_pivot_1w) if len(df_1w) > 1 else np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 2)  # Need 20 for volume MA, 2 for prior bar
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(weekly_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions: opposite Camarilla breakout (R4/S4) or weekly pivot reversal
        if position != 0:
            exit_signal = False
            # Exit long: price breaks below S4 or weekly pivot turns down
            if position == 1:
                if curr_low <= s4_1d_aligned[i]:
                    exit_signal = True
                elif curr_close < weekly_pivot_aligned[i] and prev_close >= weekly_pivot_aligned[i]:
                    exit_signal = True
            # Exit short: price breaks above R4 or weekly pivot turns up
            elif position == -1:
                if curr_high >= r4_1d_aligned[i]:
                    exit_signal = True
                elif curr_close > weekly_pivot_aligned[i] and prev_close <= weekly_pivot_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and weekly pivot trend filter
        if position == 0:
            # Camarilla breakout signals (using current bar high/low vs prior levels)
            breakout_up = curr_high >= r3_1d_aligned[i] and prev_close < r3_1d_aligned[i]
            breakout_down = curr_low <= s3_1d_aligned[i] and prev_close > s3_1d_aligned[i]
            
            # Volume confirmation: current volume > 1.8 * 20-period average volume
            volume_confirm = curr_volume > 1.8 * vol_ma_20_aligned[i]
            
            # Weekly pivot trend filter: price > weekly pivot for long, price < weekly pivot for short
            weekly_trend_up = curr_close > weekly_pivot_aligned[i]
            weekly_trend_down = curr_close < weekly_pivot_aligned[i]
            
            if breakout_up and volume_confirm and weekly_trend_up:
                signals[i] = 0.25
                position = 1
            elif breakout_down and volume_confirm and weekly_trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dVolumeSpike_1wPivotTrend_v1"
timeframe = "6h"
leverage = 1.0