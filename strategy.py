#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly and daily price action with volume confirmation.
# Weekly high/low acts as strong institutional support/resistance.
# Long when price breaks above weekly high with daily close above weekly high and volume confirmation.
# Short when price breaks below weekly low with daily close below weekly low and volume confirmation.
# Exit when price returns to weekly midpoint (mean reversion) or opposite weekly level is breached.
# Designed for low trade frequency (10-25/year) by requiring weekly breakouts with confirmation.
# Works in both bull (breakouts continue) and bear (mean reversion at extremes) markets.

name = "6h_WeeklyBreakout_DailyConfirmation_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for weekly high/low
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for daily close confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly high and low from previous week
    weekly_high = np.full_like(close_1d, np.nan)
    weekly_low = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        weekly_high[i] = high_1w[i-1]
        weekly_low[i] = low_1w[i-1]
    
    # First week has no previous data
    weekly_high[0] = weekly_low[0] = np.nan
    
    # Align weekly levels to 6h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Align daily close to 6h timeframe for confirmation
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Weekly midpoint for mean reversion exit
    weekly_midpoint = (weekly_high_aligned + weekly_low_aligned) / 2
    
    # Volume confirmation: 6h volume > 1.8x 30-period EMA (strict filter)
    vol_ema = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    vol_confirm = volume > (vol_ema * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for volume EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or
            np.isnan(daily_close_aligned[i]) or np.isnan(weekly_midpoint[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: break above weekly high with daily close confirmation and volume
            if (close[i] > weekly_high_aligned[i] and 
                daily_close_aligned[i] > weekly_high_aligned[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: break below weekly low with daily close confirmation and volume
            elif (close[i] < weekly_low_aligned[i] and 
                  daily_close_aligned[i] < weekly_low_aligned[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to weekly midpoint (mean reversion) or breaks weekly low
            if close[i] <= weekly_midpoint[i] or close[i] < weekly_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to weekly midpoint (mean reversion) or breaks weekly high
            if close[i] >= weekly_midpoint[i] or close[i] > weekly_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals