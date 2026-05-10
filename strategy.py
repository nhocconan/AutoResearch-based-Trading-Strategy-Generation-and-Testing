#!/usr/bin/env python3
# 6h_VolumeBreakout_WeeklyTrend_1dConfluence
# Hypothesis: On 6h timeframe, combine weekly trend filter (EMA34) with 1d volume spike confirmation
# and 6h price breaking above/below prior 6h high/low. This captures momentum bursts aligned with
# higher timeframe trend while avoiding chop. Volume spike ensures institutional participation.
# Works in bull markets via trend-following breaks and in bear via short breakdowns with volume.
# Target: 20-40 trades/year to minimize fee drag on 6h timeframe.

name = "6h_VolumeBreakout_WeeklyTrend_1dConfluence"
timeframe = "6h"
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
    
    # Weekly trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w_up = close_1w > ema34_1w
    trend_1w_down = close_1w < ema34_1w
    
    # Align weekly trend to 6h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # 1d volume spike confirmation (volume > 2.0 x 20-period average)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.zeros_like(volume_1d)
    vol_sum = 0
    for i in range(len(volume_1d)):
        vol_sum += volume_1d[i]
        if i >= 20:
            vol_sum -= volume_1d[i-20]
        if i >= 19:
            vol_ma_1d[i] = vol_sum / 20
        else:
            vol_ma_1d[i] = np.nan
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    
    # Align 1d volume spike to 6h
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Prior 6h high/low for breakout levels
    # Use rolling window of 2 periods to get previous bar high/low
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(prev_high[i]) or np.isnan(prev_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above prior 6h high with volume spike and weekly uptrend
            if (high[i] > prev_high[i] and
                volume_spike_1d_aligned[i] > 0.5 and
                trend_1w_up_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below prior 6h low with volume spike and weekly downtrend
            elif (low[i] < prev_low[i] and
                  volume_spike_1d_aligned[i] > 0.5 and
                  trend_1w_down_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below prior 6h low or weekly trend turns down
            if (low[i] < prev_low[i] or
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above prior 6h high or weekly trend turns up
            if (high[i] > prev_high[i] or
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals