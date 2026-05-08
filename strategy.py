#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with weekly trend filter and volume confirmation.
# Long when weekly trend is up (price > weekly EMA20), Williams %R < -80 (oversold), and volume > 1.5x 20-day average.
# Short when weekly trend is down (price < weekly EMA20), Williams %R > -20 (overbought), and volume > 1.5x 20-day average.
# Exit when Williams %R crosses back above -50 (for long) or below -50 (for short).
# Uses weekly trend to avoid counter-trend trades in strong moves, Williams %R for mean reversion entries.
# Target: 30-100 total trades over 4 years (7-25/year) with controlled frequency to avoid fee drag.

name = "1d_WilliamsR_MeanReversion_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter (EMA20)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    close_w = df_w['close'].values
    ema20_w = pd.Series(close_w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_w_aligned = align_htf_to_ltf(prices, df_w, ema20_w)
    
    # Williams %R (14-period) on daily data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_low - lowest_low + 1e-10)  # Avoid division by zero
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema20_w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: weekly uptrend, Williams %R oversold, volume confirmation
            long_cond = (close[i] > ema20_w_aligned[i]) and (williams_r[i] < -80) and volume_filter[i]
            # Short conditions: weekly downtrend, Williams %R overbought, volume confirmation
            short_cond = (close[i] < ema20_w_aligned[i]) and (williams_r[i] > -20) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -50
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals