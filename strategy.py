#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band breakout with 1w trend filter and volume confirmation.
# Long when price breaks above upper BB (1d) AND 1w EMA50 rising AND volume > 2x 20-day average.
# Short when price breaks below lower BB (1d) AND 1w EMA50 falling AND volume > 2x 20-day average.
# Exit when price crosses back inside the Bollinger Bands.
# Bollinger Bands capture volatility expansion; 1w EMA50 filters long-term trend; volume confirms institutional interest.
# Designed for 1-3 trades per month (~12-36/year) to minimize fee drag.

name = "1d_Bollinger_20_2x_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Bollinger Bands (20, 2) on 1d close
    close_1d = df_1d['close'].values
    sma20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper = sma20 + (2 * std20)
    lower = sma20 - (2 * std20)
    
    # Align Bollinger Bands to 1d timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    # 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1w EMA50 direction
    ema50_rising = np.zeros_like(ema50_1w_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_1w_aligned, dtype=bool)
    ema50_rising[1:] = ema50_1w_aligned[1:] > ema50_1w_aligned[:-1]
    ema50_falling[1:] = ema50_1w_aligned[1:] < ema50_1w_aligned[:-1]
    
    # Volume filter: current volume > 2x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 2)  # Sufficient warmup for EMA50 and Bollinger Bands
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(ema50_rising[i]) or 
            np.isnan(ema50_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper BB, 1w EMA50 rising, volume filter
            long_cond = (close[i] > upper_aligned[i]) and ema50_rising[i] and volume_filter[i]
            # Short conditions: price breaks below lower BB, 1w EMA50 falling, volume filter
            short_cond = (close[i] < lower_aligned[i]) and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below upper BB
            if close[i] < upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above lower BB
            if close[i] > lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals