#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Bollinger Band squeeze breakout with 4h trend filter and volume confirmation.
# Long when price breaks above upper BB, 4h EMA50 rising, volume > 1.5x 20-period average, and squeeze present.
# Short when price breaks below lower BB, 4h EMA50 falling, volume > 1.5x 20-period average, and squeeze present.
# Exit when price crosses back inside Bollinger Bands (middle band).
# This strategy targets low volatility breakouts with trend alignment and volume confirmation.
# Bollinger squeeze identifies periods of low volatility that often precede explosive moves.
# The 4h EMA50 filter ensures we trade with the higher timeframe trend.
# Volume confirmation filters out false breakouts.
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 4h trend direction.

name = "1h_BollingerSqueeze_Breakout_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Bollinger Bands (20, 2) on 1h
    bb_period = 20
    bb_std = 2
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_series.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_middle + (bb_std * bb_std_dev)
    bb_lower = bb_middle - (bb_std * bb_std_dev)
    
    # Bollinger Band width for squeeze detection (normalized by middle band)
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze_condition = bb_width < bb_width_ma  # True when volatility is low (squeeze)
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 4h EMA50 direction
    ema50_rising = np.zeros_like(ema50_4h_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_4h_aligned, dtype=bool)
    ema50_rising[1:] = ema50_4h_aligned[1:] > ema50_4h_aligned[:-1]
    ema50_falling[1:] = ema50_4h_aligned[1:] < ema50_4h_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 20, 50)  # Sufficient warmup for BB and EMAs
    
    for i in range(start_idx, n):
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_middle[i]) or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(ema50_rising[i]) or 
            np.isnan(ema50_falling[i]) or np.isnan(volume_filter[i]) or
            np.isnan(squeeze_condition[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper BB, 4h EMA50 rising, volume filter, squeeze present
            long_cond = (close[i] > bb_upper[i]) and ema50_rising[i] and volume_filter[i] and squeeze_condition[i]
            # Short conditions: price breaks below lower BB, 4h EMA50 falling, volume filter, squeeze present
            short_cond = (close[i] < bb_lower[i]) and ema50_falling[i] and volume_filter[i] and squeeze_condition[i]
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses back below middle BB
            if close[i] < bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses back above middle BB
            if close[i] > bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals