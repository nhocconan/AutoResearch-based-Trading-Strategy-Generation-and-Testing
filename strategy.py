#!/usr/bin/env python3
# 1D_POWER_TREND_REVERSAL_WEEKLY_CONFIRM
# Hypothesis: Daily price reversing from weekly extremes (weekly high/low) with volume
# confirmation and weekly trend alignment captures mean reversion in ranging markets
# and continuation in trending markets. Works in both bull and bear by fading
# overextended moves and catching reversals at key weekly levels.

name = "1D_POWER_TREND_REVERSAL_WEEKLY_CONFIRM"
timeframe = "1d"
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
    
    # Weekly data for extreme levels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly high and low from previous week (requires previous week's data)
    weekly_high = np.full(len(close_1w), np.nan)
    weekly_low = np.full(len(close_1w), np.nan)
    
    for i in range(1, len(close_1w)):
        weekly_high[i] = high_1w[i-1]
        weekly_low[i] = low_1w[i-1]
    
    # Weekly EMA50 for trend filter
    weekly_ema50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume spike: current volume > 2x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    # Align all weekly data to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any critical data is not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(weekly_ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price touches or goes below weekly low with volume spike in weekly uptrend
            if (low[i] <= weekly_low_aligned[i] and 
                volume_spike[i] and 
                close[i] > weekly_ema50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches or goes above weekly high with volume spike in weekly downtrend
            elif (high[i] >= weekly_high_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < weekly_ema50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above weekly EMA50 or reaches weekly high
            if (close[i] >= weekly_ema50_aligned[i] or 
                high[i] >= weekly_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below weekly EMA50 or reaches weekly low
            if (close[i] <= weekly_ema50_aligned[i] or 
                low[i] <= weekly_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals