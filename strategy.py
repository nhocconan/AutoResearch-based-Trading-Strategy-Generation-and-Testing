#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1d trend filter and volume confirmation.
# Long when Williams %R crosses above -20 (oversold bounce) AND 1d EMA50 rising AND volume > 1.5x 20-period average.
# Short when Williams %R crosses below -80 (overbought rejection) AND 1d EMA50 falling AND volume > 1.5x 20-period average.
# Exit when Williams %R crosses back below -50 for longs or above -50 for shorts.
# This strategy captures mean reversion in oversold/overbought conditions while trading with the daily trend.
# Williams %R is effective in ranging markets and during pullbacks in trends, common in 2025 BTC/ETH action.
# Volume confirmation ensures institutional participation and reduces false signals.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.

name = "4h_WilliamsR_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R (14)
    wr_length = 14
    highest_high = pd.Series(high).rolling(window=wr_length, min_periods=wr_length).max().values
    lowest_low = pd.Series(low).rolling(window=wr_length, min_periods=wr_length).min().values
    willr = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d EMA50 direction
    ema50_rising = np.zeros_like(ema50_1d_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_1d_aligned, dtype=bool)
    ema50_rising[1:] = ema50_1d_aligned[1:] > ema50_1d_aligned[:-1]
    ema50_falling[1:] = ema50_1d_aligned[1:] < ema50_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(wr_length, 50)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(willr[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_rising[i]) or np.isnan(ema50_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -20, 1d EMA50 rising, volume filter
            long_cond = (willr[i] > -20) and (willr[i-1] <= -20) and ema50_rising[i] and volume_filter[i]
            # Short conditions: Williams %R crosses below -80, 1d EMA50 falling, volume filter
            short_cond = (willr[i] < -80) and (willr[i-1] >= -80) and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses back below -50
            if willr[i] < -50 and willr[i-1] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses back above -50
            if willr[i] > -50 and willr[i-1] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals