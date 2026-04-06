#!/usr/bin/env python3
"""
6h Bollinger Band Squeeze Breakout with 1d Trend Filter
Hypothesis: Bollinger Band squeeze (low volatility) precedes breakouts. 
Breakouts in direction of 1d trend (EMA50) capture momentum moves.
Volume confirmation filters false breakouts. Works in bull/bear via trend filter.
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_bb_squeeze_breakout_1dtrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend direction
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = sma_20 + (2 * std_20)
    lower = sma_20 - (2 * std_20)
    bandwidth = (upper - lower) / sma_20  # Bollinger Band Width
    
    # Bollinger Band Squeeze: bandwidth < 20-period percentile of bandwidth
    bb_width_ma = pd.Series(bandwidth).rolling(window=20, min_periods=20).mean().values
    squeeze = bandwidth < (0.8 * bb_width_ma)  # Squeeze when BBW < 80% of its MA
    
    # 6h volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For EMA50 and BBands
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine 1d trend
        uptrend = ema_50_1d_aligned[i] > sma_20[i]  # Price above 1d EMA50 proxy
        downtrend = ema_50_1d_aligned[i] < sma_20[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: price back inside BBands OR stoploss
            if (close[i] <= sma_20[i] or  # Return to mean
                close[i] <= entry_price - 2.0 * std_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price back inside BBands OR stoploss
            if (close[i] >= sma_20[i] or  # Return to mean
                close[i] >= entry_price + 2.0 * std_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries: squeeze breakout + trend + volume
            long_breakout = squeeze[i-1] and (close[i] > upper[i-1]) and uptrend and vol_filter[i]
            short_breakout = squeeze[i-1] and (close[i] < lower[i-1]) and downtrend and vol_filter[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals