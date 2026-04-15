#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + Volume Spike + Weekly Trend Filter
# Williams %R identifies overbought/oversold conditions. Enter on reversals from extreme levels
# when confirmed by volume spike and aligned with weekly trend (using EMA50 on weekly).
# Works in ranging markets (reversals from extremes) and trending markets (pullbacks in trend).
# Target: 60-120 total trades over 4 years (15-30/year).

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
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on weekly for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Williams %R (14-period) on 6h
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume spike: current volume > 2.0 * median of last 20 periods
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median().values
    vol_spike = volume > (2.0 * vol_median)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(wr[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_median[i])):
            continue
        
        # Long entry: Williams %R crosses above -80 from oversold + volume spike + price above weekly EMA50 (uptrend)
        if (wr[i] > -80 and wr[i-1] <= -80 and  # Cross above -80
            vol_spike[i] and
            close[i] > ema50_1w_aligned[i] and   # Weekly uptrend filter
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Williams %R crosses below -20 from overbought + volume spike + price below weekly EMA50 (downtrend)
        elif (wr[i] < -20 and wr[i-1] >= -20 and  # Cross below -20
              vol_spike[i] and
              close[i] < ema50_1w_aligned[i] and  # Weekly downtrend filter
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Williams %R crosses opposite extreme or trend fails
        elif position == 1 and (wr[i] < -20 or close[i] < ema50_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (wr[i] > -80 or close[i] > ema50_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_Volume_Spike_WeeklyTrend"
timeframe = "6h"
leverage = 1.0