#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d EMA(50) pullback strategy with weekly pivot context and volume confirmation
# In trending markets, price respects EMA(50) as dynamic support/resistance
# Weekly pivot levels provide institutional context for trend strength
# Volume > 1.5x average confirms institutional participation during pullbacks
# Works in bull/bear as EMA adapts to trend and pivot bias confirms direction
# Target: 15-25 trades/year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for EMA and weekly pivot context
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    close_1d = pd.Series(df_1d['close'])
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate weekly pivot points from prior week (using 1d data)
    lookback = 5  # 5 trading days = 1 week
    if len(df_1d) < lookback:
        return np.zeros(n)
    
    # Get prior week's OHLC (excluding current incomplete day)
    prev_week_high = pd.Series(df_1d['high']).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    prev_week_low = pd.Series(df_1d['low']).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    prev_week_close = pd.Series(df_1d['close']).rolling(window=lookback, min_periods=lookback).last().shift(1).values
    
    # Weekly pivot calculation (standard floor trader pivot)
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    weekly_r1 = 2 * weekly_pivot - prev_week_low
    weekly_s1 = 2 * weekly_pivot - prev_week_high
    
    # Align EMA and weekly pivot levels to 1d timeframe (already aligned, but use for consistency)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, 50)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend bias: price relative to EMA(50)
        above_ema = close[i] > ema_50_aligned[i]
        below_ema = close[i] < ema_50_aligned[i]
        
        # Pivot bias: price relative to weekly pivot
        above_pivot = close[i] > weekly_pivot_aligned[i]
        below_pivot = close[i] < weekly_pivot_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: pullback to EMA(50) in uptrend (price > EMA and > pivot) + volume
            if (above_ema and 
                above_pivot and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: pullback to EMA(50) in downtrend (price < EMA and < pivot) + volume
            elif (below_ema and 
                  below_pivot and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below EMA(50) or returns to weekly pivot
            if close[i] < ema_50_aligned[i] or close[i] < weekly_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above EMA(50) or returns to weekly pivot
            if close[i] > ema_50_aligned[i] or close[i] > weekly_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_EMA50_Pullback_WeeklyPivot_Volume_v1"
timeframe = "1d"
leverage = 1.0