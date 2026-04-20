#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 20-week Donchian channels (long-term breakout levels)
    highest_20w = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_20w = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Calculate 50-day EMA of daily close (intermediate trend filter)
    ema_50d_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly and daily indicators to 4h timeframe
    highest_20w_aligned = align_htf_to_ltf(prices, df_1w, highest_20w)
    lowest_20w_aligned = align_htf_to_ltf(prices, df_1w, lowest_20w)
    ema_50d_aligned = align_htf_to_ltf(prices, df_1d, ema_50d_1d)
    
    # Calculate 4h ATR for volatility filter and stop sizing
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Precompute hour of day for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Volume filter: 4h volume > 20-period average
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if NaN in indicators
        if np.isnan(highest_20w_aligned[i]) or np.isnan(lowest_20w_aligned[i]) or np.isnan(ema_50d_aligned[i]) or np.isnan(atr_4h[i]):
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
        resistance = highest_20w_aligned[i]
        support = lowest_20w_aligned[i]
        trend_filter = ema_50d_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above 20-week resistance, above 50-day EMA, with volume
            if price > resistance and price > trend_filter and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below 20-week support, below 50-day EMA, with volume
            elif price < support and price < trend_filter and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss (2x ATR below entry) or price breaks below 20-week support
            if price <= entry_price - 2.0 * atr_4h[i] or price < support:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss (2x ATR above entry) or price breaks above 20-week resistance
            if price >= entry_price + 2.0 * atr_4h[i] or price > resistance:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_20W_Donchian_EMA50_Trend_VolumeFilter"
timeframe = "4h"
leverage = 1.0