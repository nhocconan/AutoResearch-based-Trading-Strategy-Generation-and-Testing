#!/usr/bin/env python3
"""
6h_1d_WeeklyPivot_Donchian20_Breakout_Volume
Hypothesis: Combine weekly pivot levels (Monday's weekly high/low) with daily Donchian(20) breakout and volume confirmation on 6h timeframe. Weekly pivot defines the weekly bias, Donchian breakout provides entry timing, and volume confirms momentum. Works in bull markets by buying breakouts above weekly pivot resistance, and in bear markets by selling breakdowns below weekly pivot support. Targets 15-30 trades/year by requiring alignment of weekly bias, Donchian breakout, and volume > 2x 20-period average.
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
    
    # Get daily data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot (Monday's weekly high/low)
    # Convert to pandas Series for resampling
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    close_series = pd.Series(close_1d)
    
    # Resample to weekly (Monday start)
    weekly_high = high_series.resample('W-MON', label='left').max().values
    weekly_low = low_series.resample('W-MON', label='left').min().values
    weekly_close = close_series.resample('W-MON', label='left').last().values
    
    # Align weekly data to daily index first
    # Create dummy index for weekly data
    weekly_index = pd.date_range(start=df_1d.index[0], periods=len(weekly_high), freq='W-MON')
    # Reindex to daily
    weekly_high_daily = pd.Series(weekly_high, index=weekly_index).reindex(df_1d.index, method='ffill').values
    weekly_low_daily = pd.Series(weekly_low, index=weekly_index).reindex(df_1d.index, method='ffill').values
    
    # Align weekly pivot to 6h timeframe
    weekly_high_6h = align_htf_to_ltf(prices, df_1d, weekly_high_daily)
    weekly_low_6h = align_htf_to_ltf(prices, df_1d, weekly_low_daily)
    
    # Get 6h Donchian(20) channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_high_6h[i]) or np.isnan(weekly_low_6h[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above weekly high AND Donchian high, with volume
            if (close[i] > weekly_high_6h[i] and close[i] > donchian_high[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly low AND Donchian low, with volume
            elif (close[i] < weekly_low_6h[i] and close[i] < donchian_low[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns below weekly low or Donchian low
            if (close[i] < weekly_low_6h[i] or close[i] < donchian_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above weekly high or Donchian high
            if (close[i] > weekly_high_6h[i] or close[i] > donchian_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_WeeklyPivot_Donchian20_Breakout_Volume"
timeframe = "6h"
leverage = 1.0