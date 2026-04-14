#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points using prior week's OHLC (approximated with 5-day lookback)
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(1).values
    
    # Weekly pivot point: (H + L + C) / 3
    pp = (weekly_high + weekly_low + weekly_close) / 3
    # Resistance levels
    r1 = 2 * pp - weekly_low
    r2 = pp + (weekly_high - weekly_low)
    # Support levels
    s1 = 2 * pp - weekly_high
    s2 = pp - (weekly_high - weekly_low)
    
    # Align pivot levels to 4h timeframe (wait for daily bar close)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)  # 50 for weekly EMA, 20 for volume
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above R2 resistance AND above weekly EMA50 with volume
            if price > r2_aligned[i] and price > ema_50_1w_aligned[i] and vol > 1.5 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below S2 support AND below weekly EMA50 with volume
            elif price < s2_aligned[i] and price < ema_50_1w_aligned[i] and vol > 1.5 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S2 support
            if price < s2_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above R2 resistance
            if price > r2_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1w_Weekly_Pivot_Breakout_Trend"
timeframe = "4h"
leverage = 1.0