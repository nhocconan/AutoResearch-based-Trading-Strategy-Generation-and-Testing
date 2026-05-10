# 6h_WeeklyPivot_VolumeRegime_Signal
# Hypothesis: Uses weekly pivot points from 1w data with volume regime filter and 1d trend filter.
# Long when price breaks above weekly R1 with volume expansion and 1d uptrend.
# Short when price breaks below weekly S1 with volume expansion and 1d downtrend.
# Weekly pivots provide structural support/resistance that works in both bull and bear markets.
# Volume regime filter ensures trades occur during high conviction moves.
# Target: 15-30 trades/year to avoid fee drag on 6h timeframe.

name = "6h_WeeklyPivot_VolumeRegime_Signal"
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
    
    # Get 1w data for weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week's OHLC
    prev_week_close = df_1w['close'].shift(1).values
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    
    # Weekly pivot point calculation
    pp = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = 2 * pp - prev_week_low
    s1 = 2 * pp - prev_week_high
    
    # Align weekly pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Get 1d data for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume regime filter: volume > 1.5x 20-period MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Warmup for volume MA and 1d EMA
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 1d
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume regime filter
        volume_regime = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: price breaks above weekly R1 with volume expansion and 1d uptrend
            if close[i] > r1_aligned[i] and volume_regime and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly S1 with volume expansion and 1d downtrend
            elif close[i] < s1_aligned[i] and volume_regime and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls back below weekly R1 or trend turns down
            if close[i] < r1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises back above weekly S1 or trend turns up
            if close[i] > s1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals