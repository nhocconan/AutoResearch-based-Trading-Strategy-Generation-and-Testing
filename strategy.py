#!/usr/bin/env python3
# 1d_1w_TRIX_PriceChannel_Breakout
# Hypothesis: Uses weekly TRIX momentum to determine trend direction and daily price channel breakouts for entry.
# In bull markets: weekly TRIX positive + daily breakout above upper channel = long.
# In bear markets: weekly TRIX negative + daily breakout below lower channel = short.
# Volume confirmation ensures breakouts have conviction. Target: 10-25 trades/year to minimize fee drag.

name = "1d_1w_TRIX_PriceChannel_Breakout"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for TRIX calculation
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly TRIX (15-period EMA of EMA of EMA of close) ---
    close_1w = df_1w['close'].values
    ema1 = pd.Series(close_1w).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = pd.Series(ema3).pct_change(periods=1) * 100  # Percentage change
    trix_1w = trix.values
    trix_1w_aligned = align_htf_to_ltf(prices, df_1w, trix_1w)
    
    # --- Daily Price Channel (Donchian 20) ---
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- Volume confirmation (1.5x 20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for TRIX (45 periods) and Donchian (20 periods)
    start_idx = 45
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix_1w_aligned[i]) or
            np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: weekly TRIX positive + price breaks above upper channel + volume surge
            if (trix_1w_aligned[i] > 0 and 
                close[i] > high_20[i] and 
                volume_surge):
                signals[i] = 0.25
                position = 1
            # Short: weekly TRIX negative + price breaks below lower channel + volume surge
            elif (trix_1w_aligned[i] < 0 and 
                  close[i] < low_20[i] and 
                  volume_surge):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below lower channel OR weekly TRIX turns negative
                if (close[i] < low_20[i] or 
                    trix_1w_aligned[i] < 0):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises above upper channel OR weekly TRIX turns positive
                if (close[i] > high_20[i] or 
                    trix_1w_aligned[i] > 0):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals