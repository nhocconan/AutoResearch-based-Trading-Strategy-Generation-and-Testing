#!/usr/bin/env python3
"""
6h_WeeklyPivot_DailyTrend_VolumeBreakout
Hypothesis: Uses weekly pivot levels (R1/S1) as dynamic support/resistance, with breakouts confirmed by daily trend and volume spikes. Works in bull markets by buying breakouts above weekly R1 in uptrend, and in bear markets by selling breakdowns below weekly S1 in downtrend. Targets 15-30 trades/year per symbol.
"""

name = "6h_WeeklyPivot_DailyTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # Weekly pivot points (using weekly high/low/close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot: (H + L + C) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low  # Resistance 1
    weekly_s1 = 2 * weekly_pivot - weekly_high  # Support 1
    
    # Align weekly levels to 6h (weekly pivot updates only when weekly candle closes)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1, additional_delay_bars=0)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1, additional_delay_bars=0)
    
    # Daily trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align daily trend to 6h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # Volume confirmation: 20-period average
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 2.0  # Require strong volume spike
        
        if position == 0:
            # Enter long: price breaks above weekly R1 + daily uptrend + volume spike
            if (close[i] > weekly_r1_aligned[i] and 
                trend_1d_up_aligned[i] > 0.5 and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly S1 + daily downtrend + volume spike
            elif (close[i] < weekly_s1_aligned[i] and 
                  trend_1d_down_aligned[i] > 0.5 and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price falls back below weekly pivot or trend changes
            if (close[i] < weekly_pivot[i] if not np.isnan(weekly_pivot[i]) else False) or \
               trend_1d_up_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price rises back above weekly pivot or trend changes
            if (close[i] > weekly_pivot[i] if not np.isnan(weekly_pivot[i]) else False) or \
               trend_1d_down_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Note: weekly_pivot array used in exit conditions - need to compute and align it
# Re-computing weekly_pivot for exit conditions
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot, additional_delay_bars=0)