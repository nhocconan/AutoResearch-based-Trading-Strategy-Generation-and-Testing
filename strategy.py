#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 10-day Donchian breakout with 1-day volume confirmation and 1-week trend filter
# Long when price breaks above 10-day high AND above 1-week low (uptrend) AND volume > 20-period average
# Short when price breaks below 10-day low AND below 1-week high (downtrend) AND volume > 20-period average
# Exit when price reverses back across the 10-day level or stop loss (2x ATR) is hit
# Target: 20-50 total trades over 4 years (5-12/year) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load weekly data ONCE for 1-week trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1-week high/low (trend filter)
    highest_1w = pd.Series(high_1w).rolling(window=1, min_periods=1).max().values  # current week high
    lowest_1w = pd.Series(low_1w).rolling(window=1, min_periods=1).min().values    # current week low
    
    # Align weekly trend filters to 4h timeframe
    highest_1w_aligned = align_htf_to_ltf(prices, df_1w, highest_1w)
    lowest_1w_aligned = align_htf_to_ltf(prices, df_1w, lowest_1w)
    
    # Load daily data ONCE for 10-day Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 10-day Donchian channels (breakout levels)
    highest_10d = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    lowest_10d = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    
    # Align daily breakout levels to 4h timeframe
    highest_10d_aligned = align_htf_to_ltf(prices, df_1d, highest_10d)
    lowest_10d_aligned = align_htf_to_ltf(prices, df_1d, lowest_10d)
    
    # Calculate 4h ATR for volatility filter and stop sizing
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Precompute hour of day for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Volume filter: 4h volume > 20-period average
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if NaN in indicators
        if np.isnan(highest_10d_aligned[i]) or np.isnan(lowest_10d_aligned[i]) or \
           np.isnan(highest_1w_aligned[i]) or np.isnan(lowest_1w_aligned[i]) or np.isnan(atr_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter
        vol_filter = volume[i] > volume_ma_20[i]
        
        # Price levels
        resistance = highest_10d_aligned[i]
        support = lowest_10d_aligned[i]
        weekly_high = highest_1w_aligned[i]
        weekly_low = lowest_1w_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above 10-day resistance, above weekly low (uptrend), with volume
            if price > resistance and price > weekly_low and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below 10-day support, below weekly high (downtrend), with volume
            elif price < support and price < weekly_high and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss (2x ATR below entry) or price breaks below 10-day support
            if price <= entry_price - 2.0 * atr_4h[i] or price < support:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss (2x ATR above entry) or price breaks above 10-day resistance
            if price >= entry_price + 2.0 * atr_4h[i] or price > resistance:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_10D_Donchian_1W_Trend_VolumeFilter"
timeframe = "4h"
leverage = 1.0