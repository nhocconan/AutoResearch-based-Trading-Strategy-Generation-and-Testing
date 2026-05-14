#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 12h trend filter and Bollinger Band mean reversion
# Strategy: In strong trends (price > 12h EMA50), look for mean reversion to Bollinger Band middle (20 SMA)
# In ranging markets (price near Bollinger middle), follow 12h trend direction
# This adapts to both bull/bear markets by using trend-aware mean reversion
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for trend and Bollinger Bands
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h Bollinger Bands (20, 2)
    sma_20_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    std_20_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    upper_bb_12h = sma_20_12h + 2 * std_20_12h
    lower_bb_12h = sma_20_12h - 2 * std_20_12h
    
    # Align 12h indicators to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, sma_20_12h)
    upper_bb_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_bb_12h)
    lower_bb_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_bb_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(sma_20_12h_aligned[i]) or 
            np.isnan(upper_bb_12h_aligned[i]) or 
            np.isnan(lower_bb_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine market regime based on price vs Bollinger Bands
        price_vs_upper = (close[i] - upper_bb_12h_aligned[i]) / (upper_bb_12h_aligned[i] - sma_20_12h_aligned[i] + 1e-10)
        price_vs_lower = (close[i] - lower_bb_12h_aligned[i]) / (sma_20_12h_aligned[i] - lower_bb_12h_aligned[i] + 1e-10)
        
        # Strong trend: price outside Bollinger Bands
        strong_uptrend = price_vs_upper > 0 and close[i] > ema_50_12h_aligned[i]
        strong_downtrend = price_vs_lower < 0 and close[i] < ema_50_12h_aligned[i]
        
        # Ranging: price near Bollinger middle (within 0.5 bands)
        near_middle = abs(close[i] - sma_20_12h_aligned[i]) < 0.5 * (upper_bb_12h_aligned[i] - lower_bb_12h_aligned[i])
        
        # Entry logic
        long_entry = False
        short_entry = False
        
        if strong_uptrend:
            # In strong uptrend, look for pullbacks to middle (mean reversion)
            if near_middle and close[i] > sma_20_12h_aligned[i]:
                long_entry = True
        elif strong_downtrend:
            # In strong downtrend, look for bounces to middle
            if near_middle and close[i] < sma_20_12h_aligned[i]:
                short_entry = True
        elif near_middle:
            # In ranging market, follow 12h trend
            if close[i] > ema_50_12h_aligned[i]:
                long_entry = True
            elif close[i] < ema_50_12h_aligned[i]:
                short_entry = True
        
        # Exit conditions: opposite signal or break of Bollinger Band
        exit_long = position == 1 and (close[i] < lower_bb_12h_aligned[i] or short_entry)
        exit_short = position == -1 and (close[i] > upper_bb_12h_aligned[i] or long_entry)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_bollinger_trend_mean_reversion_v1"
timeframe = "6h"
leverage = 1.0