#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-day/1-week Choppiness index + ATR-based Donchian breakout.
# Uses weekly trend from Choppiness index to filter direction and daily ATR/Donchian for entries.
# Designed to reduce whipsaw in sideways markets and capture trends in both bull and bear regimes.
# Target: 20-40 trades per year with 0.25 position size to manage drawdown.

name = "12h_1w1d_chop_donchian_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly ATR for Choppiness input
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    atr_period = 14
    atr_1w = np.full_like(tr, np.nan, dtype=float)
    for i in range(atr_period-1, len(tr)):
        if i == atr_period-1:
            atr_1w[i] = np.mean(tr[:i+1])
        else:
            atr_1w[i] = (atr_1w[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate weekly Choppiness index
    atr_sum = np.full_like(tr, np.nan, dtype=float)
    for i in range(atr_period-1, len(tr)):
        atr_sum[i] = np.sum(tr[i-atr_period+1:i+1])
    
    hh = np.full_like(tr, np.nan, dtype=float)
    ll = np.full_like(tr, np.nan, dtype=float)
    for i in range(atr_period-1, len(tr)):
        hh[i] = np.max(high_1w[i-atr_period+1:i+1])
        ll[i] = np.min(low_1w[i-atr_period+1:i+1])
    
    chop = np.full_like(tr, np.nan, dtype=float)
    for i in range(atr_period-1, len(tr)):
        if hh[i] != ll[i] and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(atr_period)
        else:
            chop[i] = 50.0
    
    # Weekly trend: chop > 61.8 = range (mean revert), chop < 38.2 = trending
    chop_threshold_high = 61.8
    chop_threshold_low = 38.2
    chop_range = chop > chop_threshold_high
    chop_trend = chop < chop_threshold_low
    
    # Align chop regimes to 12h
    chop_range_aligned = align_htf_to_ltf(prices, df_1w, chop_range)
    chop_trend_aligned = align_htf_to_ltf(prices, df_1w, chop_trend)
    
    # Calculate daily ATR and Donchian channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1_d = high_1d - low_1d
    tr2_d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    tr_d[0] = tr1_d[0]
    
    atr_period_d = 10
    atr_1d = np.full_like(tr_d, np.nan, dtype=float)
    for i in range(atr_period_d-1, len(tr_d)):
        if i == atr_period_d-1:
            atr_1d[i] = np.mean(tr_d[:i+1])
        else:
            atr_1d[i] = (atr_1d[i-1] * (atr_period_d-1) + tr_d[i]) / atr_period_d
    
    donchian_period = 20
    highest_high = np.full_like(high_1d, np.nan, dtype=float)
    lowest_low = np.full_like(low_1d, np.nan, dtype=float)
    for i in range(donchian_period-1, len(high_1d)):
        highest_high[i] = np.max(high_1d[i-donchian_period+1:i+1])
        lowest_low[i] = np.min(low_1d[i-donchian_period+1:i+1])
    
    # Align daily indicators to 12h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    highest_high_aligned = align_htf_to_ltf(prices, df_1d, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1d, lowest_low)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(highest_high_aligned[i]) or 
            np.isnan(lowest_low_aligned[i]) or
            np.isnan(chop_range_aligned[i]) or np.isnan(chop_trend_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # In ranging markets: mean revert at Donchian extremes
        # In trending markets: breakout in direction of trend
        if chop_range_aligned[i]:
            # Range: fade at Donchian bands
            fade_long = low[i] <= lowest_low_aligned[i]
            fade_short = high[i] >= highest_high_aligned[i]
            
            # Exit when price returns to middle of channel
            mid_point = (highest_high_aligned[i] + lowest_low_aligned[i]) / 2
            exit_long = position == 1 and close[i] >= mid_point
            exit_short = position == -1 and close[i] <= mid_point
            
            if fade_long and position != 1:
                position = 1
                signals[i] = 0.25
            elif fade_short and position != -1:
                position = -1
                signals[i] = -0.25
            elif position == 1 and exit_long:
                position = 0
                signals[i] = 0.0
            elif position == -1 and exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # Trend: breakout in direction of price action
            breakout_long = high[i] >= highest_high_aligned[i]
            breakout_short = low[i] <= lowest_low_aligned[i]
            
            # Exit on opposite Donchian touch
            exit_long = position == 1 and low[i] <= lowest_low_aligned[i]
            exit_short = position == -1 and high[i] >= highest_high_aligned[i]
            
            if breakout_long and position != 1:
                position = 1
                signals[i] = 0.25
            elif breakout_short and position != -1:
                position = -1
                signals[i] = -0.25
            elif position == 1 and exit_long:
                position = 0
                signals[i] = 0.0
            elif position == -1 and exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals