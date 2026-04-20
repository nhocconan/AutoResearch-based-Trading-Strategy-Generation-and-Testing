#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 10-day highest high and lowest low (short-term range)
    highest_10d = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    lowest_10d = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    
    # Align daily levels to 4h timeframe
    highest_10d_aligned = align_htf_to_ltf(prices, df_1d, highest_10d)
    lowest_10d_aligned = align_htf_to_ltf(prices, df_1d, lowest_10d)
    
    # Calculate 20-day EMA of daily close (long-term trend filter)
    ema_20d_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20d_aligned = align_htf_to_ltf(prices, df_1d, ema_20d_1d)
    
    # Calculate 4h ATR for volatility filter and stop sizing
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=15, min_periods=15).mean().values
    
    # Calculate 2x ATR stop level
    atr_stop = 2 * atr_4h
    
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
        if np.isnan(highest_10d_aligned[i]) or np.isnan(lowest_10d_aligned[i]) or np.isnan(ema_20d_aligned[i]) or np.isnan(atr_4h[i]):
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
        trend_filter = ema_20d_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above 10-day resistance, above 20-day EMA, with volume
            if price > resistance and price > trend_filter and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below 10-day support, below 20-day EMA, with volume
            elif price < support and price < trend_filter and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss (2x ATR below entry) or price breaks below 10-day support
            if price <= entry_price - atr_stop[i] or price < support:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss (2x ATR above entry) or price breaks above 10-day resistance
            if price >= entry_price + atr_stop[i] or price > resistance:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_10D_HighLow_EMA20Trend_VolumeFilter_ATRStop"
timeframe = "4h"
leverage = 1.0