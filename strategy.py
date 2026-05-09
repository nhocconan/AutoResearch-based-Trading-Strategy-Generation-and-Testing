#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyTrend_DailyBreakout_Volume"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly EMA50 trend
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily Donchian channels (20)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data unavailable
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ema20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above 20-day high with volume filter and above weekly EMA50
            if price > high_20[i] and vol_filter[i] and price > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low with volume filter and below weekly EMA50
            elif price < low_20[i] and vol_filter[i] and price < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 20-day low (mean reversion)
            if price < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 20-day high (mean reversion)
            if price > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals