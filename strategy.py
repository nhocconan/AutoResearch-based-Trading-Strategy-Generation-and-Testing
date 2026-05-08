#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TripleMA_Crossover_Volume_Trend_v1"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMAs on 4h timeframe
    close_series = pd.Series(close)
    ema_fast = close_series.ewm(span=9, adjust=False, min_periods=9).values
    ema_medium = close_series.ewm(span=21, adjust=False, min_periods=21).values
    ema_slow = close_series.ewm(span=55, adjust=False, min_periods=55).values
    
    # Daily EMA for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 55  # Wait for slow EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_fast[i]) or np.isnan(ema_medium[i]) or np.isnan(ema_slow[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish crossover: fast > medium > slow AND above daily EMA + volume
            if (ema_fast[i] > ema_medium[i] > ema_slow[i] and 
                close[i] > ema_34_1d_aligned[i] and
                vol_ratio[i] > 1.3):
                signals[i] = 0.25
                position = 1
            # Bearish crossover: fast < medium < slow AND below daily EMA + volume
            elif (ema_fast[i] < ema_medium[i] < ema_slow[i] and 
                  close[i] < ema_34_1d_aligned[i] and
                  vol_ratio[i] > 1.3):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish crossover OR below daily EMA
            if (ema_fast[i] < ema_medium[i] or close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish crossover OR above daily EMA
            if (ema_fast[i] > ema_medium[i] or close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals