#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index + 1d EMA34 + Volume Spike
# Long when CHOP(12h) > 61.8 (range) AND price > 1d EMA34 (up trend) AND volume > 1.5x 20-period average
# Short when CHOP(12h) > 61.8 (range) AND price < 1d EMA34 (down trend) AND volume > 1.5x 20-period average
# Exit when CHOP(12h) < 38.2 (trending) OR price crosses EMA34 in opposite direction
# Choppiness Index identifies ranging markets ideal for mean reversion to EMA.
# EMA34 filters higher timeframe trend direction. Volume confirms institutional participation.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Chop_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # 1d EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 12h Choppiness Index (14-period)
    # CHOP = 100 * log10( sum(ATR) / (max(high) - min(low)) ) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low
    
    chop = np.full(n, 50.0)  # default neutral
    valid = (range_hl > 0) & ~np.isnan(atr_sum)
    chop[valid] = 100 * np.log10(atr_sum[valid] / range_hl[valid]) / np.log10(14)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20, 14)  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: ranging market (CHOP > 61.8), price above EMA34, volume filter
            long_cond = (chop[i] > 61.8) and (close[i] > ema34_1d_aligned[i]) and volume_filter[i]
            # Short conditions: ranging market (CHOP > 61.8), price below EMA34, volume filter
            short_cond = (chop[i] > 61.8) and (close[i] < ema34_1d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trending market (CHOP < 38.2) OR price crosses below EMA34
            if (chop[i] < 38.2) or (close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trending market (CHOP < 38.2) OR price crosses above EMA34
            if (chop[i] < 38.2) or (close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals