#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE for 1w indicators
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 10-week Donchian channels (breakout levels)
    highest_10w = pd.Series(high_1w).rolling(window=10, min_periods=10).max().values
    lowest_10w = pd.Series(low_1w).rolling(window=10, min_periods=10).min().values
    
    # Calculate 20-week EMA of weekly close (long-term trend filter)
    ema_20w_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all 1w indicators to 12h timeframe
    highest_10w_aligned = align_htf_to_ltf(prices, df_1w, highest_10w)
    lowest_10w_aligned = align_htf_to_ltf(prices, df_1w, lowest_10w)
    ema_20w_aligned = align_htf_to_ltf(prices, df_1w, ema_20w_1w)
    
    # Calculate 12h ATR for volatility filter and stop sizing
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Precompute hour of day for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Volume filter: 12h volume > 20-period average
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(highest_10w_aligned[i]) or np.isnan(lowest_10w_aligned[i]) or np.isnan(ema_20w_aligned[i]) or np.isnan(atr_12h[i]):
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
        resistance = highest_10w_aligned[i]
        support = lowest_10w_aligned[i]
        trend_filter = ema_20w_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above 10-week resistance, above 20-week EMA, with volume
            if price > resistance and price > trend_filter and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below 10-week support, below 20-week EMA, with volume
            elif price < support and price < trend_filter and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss (2x ATR below entry) or price breaks below 10-week support
            if price <= entry_price - 2.0 * atr_12h[i] or price < support:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss (2x ATR above entry) or price breaks above 10-week resistance
            if price >= entry_price + 2.0 * atr_12h[i] or price > resistance:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_10W_Donchian_EMA20_Trend_VolumeFilter"
timeframe = "12h"
leverage = 1.0