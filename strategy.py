#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + Volume Spike + Weekly Trend Filter
# Uses Williams %R(14) for mean reversion entries in overbought/oversold conditions.
# Entry conditions: Williams %R < -80 (oversold) or > -20 (overbought) + volume spike (2x median) + weekly trend alignment.
# Weekly trend: price above/below weekly 20-period EMA. Works in both bull and bear markets by fading extremes in the direction of the weekly trend.
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate weekly 20-period EMA
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(20, n):  # Start after Williams %R warmup
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_20_1w_aligned[i])):
            continue
        
        # Volume spike: current volume > 2x median of past 20 periods
        vol_median = np.median(volume[max(0, i-20):i+1])
        volume_spike = volume[i] > 2 * vol_median
        
        # Long entry: oversold + volume spike + price above weekly EMA (uptrend)
        if (williams_r[i] < -80 and
            volume_spike and
            close[i] > ema_20_1w_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: overbought + volume spike + price below weekly EMA (downtrend)
        elif (williams_r[i] > -20 and
              volume_spike and
              close[i] < ema_20_1w_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Williams %R returns to neutral range (-50) or opposite extreme
        elif position == 1 and williams_r[i] > -50:
            position = 0
            signals[i] = 0.0
        elif position == -1 and williams_r[i] < -50:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_Volume_WeeklyTrend"
timeframe = "6h"
leverage = 1.0