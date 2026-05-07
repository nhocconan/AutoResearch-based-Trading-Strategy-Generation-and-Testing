#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) AND 1d EMA34 rising AND volume > 1.5x 20-period average.
# Short when Williams %R > -20 (overbought) AND 1d EMA34 falling AND volume > 1.5x 20-period average.
# Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts).
# Williams %R identifies exhaustion points in trends, effective in both bull and bear markets.
# The 1d EMA34 filter ensures trades align with the daily trend, reducing counter-trend whipsaws.
# Volume confirmation ensures institutional participation and reduces false signals.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.

name = "6h_WilliamsR_1dEMA34_Volume"
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
    
    # Williams %R (14 periods)
    wr_length = 14
    highest_high = pd.Series(high).rolling(window=wr_length, min_periods=wr_length).max().values
    lowest_low = pd.Series(low).rolling(window=wr_length, min_periods=wr_length).min().values
    # Avoid division by zero
    wr_numerator = highest_high - close
    wr_denominator = highest_high - lowest_low
    wr_denominator_safe = np.where(wr_denominator == 0, 1, wr_denominator)
    williams_r = -100 * (wr_numerator / wr_denominator_safe)
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(wr_length, 34)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(williams_r[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_rising[i]) or np.isnan(ema34_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R oversold (< -80), 1d EMA34 rising, volume filter
            long_cond = (williams_r[i] < -80) and ema34_rising[i] and volume_filter[i]
            # Short conditions: Williams %R overbought (> -20), 1d EMA34 falling, volume filter
            short_cond = (williams_r[i] > -20) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses back above -50 (exiting oversold)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses back below -50 (exiting overbought)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals