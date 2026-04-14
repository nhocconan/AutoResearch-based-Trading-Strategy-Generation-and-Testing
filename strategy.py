#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h/12h/1d Trend Alignment with Volume Confirmation
# Uses 12h EMA21 for trend direction, 1d EMA50 for higher timeframe filter,
# and volume spike on 6h for entry confirmation. Only takes long when
# both 12h and 1d trends are up and volume spikes, short when both down.
# Designed to capture strong trends with minimal whipsaw by requiring
# multi-timeframe alignment. Target: 50-120 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h EMA21 for trend
    close_12h = df_12h['close'].values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60  # for EMA calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_21_12h[i]) or np.isnan(ema_50_1d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Align 12h EMA21 to 6h
        ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
        
        # Align 1d EMA50 to 6h
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
        
        price = close[i]
        
        if position == 0:
            # Long setup: both 12h and 1d trends up with volume spike
            if (price > ema_21_12h_aligned[i] and    # Above 12h EMA21 (uptrend)
                price > ema_50_1d_aligned[i] and     # Above 1d EMA50 (higher timeframe uptrend)
                vol_spike[i]):                       # Volume confirmation
                position = 1
                signals[i] = position_size
            # Short setup: both 12h and 1d trends down with volume spike
            elif (price < ema_21_12h_aligned[i] and  # Below 12h EMA21 (downtrend)
                  price < ema_50_1d_aligned[i] and   # Below 1d EMA50 (higher timeframe downtrend)
                  vol_spike[i]):                     # Volume confirmation
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below 12h EMA21 or 1d EMA50
            if price < ema_21_12h_aligned[i] or price < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above 12h EMA21 or 1d EMA50
            if price > ema_21_12h_aligned[i] or price > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_1d_EMA_Trend_Alignment_Volume"
timeframe = "6h"
leverage = 1.0