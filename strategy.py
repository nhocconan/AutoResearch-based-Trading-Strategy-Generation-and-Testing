#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Squeeze Breakout with Volume Spike and 12h EMA Trend Filter
# Uses Bollinger Bands squeeze (low volatility) as a precursor to explosive moves.
# Entry on breakout of Bollinger Bands with volume spike confirmation.
# Trend filter uses 12h EMA50 to ensure we trade in direction of higher timeframe trend.
# Works in bull markets (breakouts up with trend) and bear markets (breakouts down with trend).
# Target: 50-150 total trades over 4 years to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Bollinger Bands (20, 2.0) on daily
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std()
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Squeeze condition: Bollinger Band Width < 20th percentile of last 50 days
    bb_width = upper_bb - lower_bb
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).quantile(0.2)
    squeeze = bb_width < bb_width_percentile.values
    
    # Breakout conditions
    breakout_up = close_1d > upper_bb
    breakout_down = close_1d < lower_bb
    
    # Align signals to 4h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze.values.astype(float))
    breakout_up_aligned = align_htf_to_ltf(prices, df_1d, breakout_up.values.astype(float))
    breakout_down_aligned = align_htf_to_ltf(prices, df_1d, breakout_down.values.astype(float))
    
    # EMA50 on 12h for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(squeeze_aligned[i]) or np.isnan(breakout_up_aligned[i]) or 
            np.isnan(breakout_down_aligned[i]) or np.isnan(ema_50_aligned[i])):
            continue
        
        # Long entry: squeeze breakout up + volume spike + price above 12h EMA50
        if (squeeze_aligned[i] == 1 and breakout_up_aligned[i] == 1 and
            volume[i] > 2.0 * np.median(window := volume[max(0, i-20):i+1]) and
            close[i] > ema_50_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: squeeze breakout down + volume spike + price below 12h EMA50
        elif (squeeze_aligned[i] == 1 and breakout_down_aligned[i] == 1 and
              volume[i] > 2.0 * np.median(window := volume[max(0, i-20):i+1]) and
              close[i] < ema_50_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite breakout or volatility expansion (end of squeeze)
        elif position == 1 and (breakout_down_aligned[i] == 1 or squeeze_aligned[i] == 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (breakout_up_aligned[i] == 1 or squeeze_aligned[i] == 0):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Bollinger_Squeeze_Volume_EMA50"
timeframe = "4h"
leverage = 1.0