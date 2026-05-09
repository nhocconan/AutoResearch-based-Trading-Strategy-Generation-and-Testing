#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_Trend_Follow_With_Volume_Confirmation"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute hours for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # 4h trend: EMA20
    df_4h = get_htf_data(prices, '4h')
    ema20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1h volume filter: volume > 1.5 * 20-period SMA of volume
    vol_sma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_sma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # enough for EMA50 on 1d
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if required data unavailable
        if np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or \
           np.isnan(vol_sma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price above 4h EMA20 and 1d EMA50 + volume confirmation
            if (price > ema20_4h_aligned[i] and
                price > ema50_1d_aligned[i] and
                vol_filter[i]):
                signals[i] = 0.20
                position = 1
                continue
            
            # Short: price below 4h EMA20 and 1d EMA50 + volume confirmation
            elif (price < ema20_4h_aligned[i] and
                  price < ema50_1d_aligned[i] and
                  vol_filter[i]):
                signals[i] = -0.20
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price below 4h EMA20 or 1d EMA50
            if (price < ema20_4h_aligned[i] or
                price < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price above 4h EMA20 or 1d EMA50
            if (price > ema20_4h_aligned[i] or
                price > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals