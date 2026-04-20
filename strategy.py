#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE for HTF regime
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Donchian channels (20-period)
    highest_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily ATR(14) for volatility and stop
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily indicators to 4h timeframe
    highest_high_20_aligned = align_htf_to_ltf(prices, df_1d, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_20)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Precompute hour of day for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # 4h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in HTF indicators
        if np.isnan(highest_high_20_aligned[i]) or np.isnan(lowest_low_20_aligned[i]) or np.isnan(atr_14_1d_aligned[i]):
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
        
        hh_20 = highest_high_20_aligned[i]
        ll_20 = lowest_low_20_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: current volume must be above 50-period average (use 4h volume)
        if i >= 50:
            vol_ma_50 = np.mean(volume[i-50:i])
            vol_filter = vol > vol_ma_50
        else:
            vol_filter = False
        
        if position == 0:
            # Long: price breaks above 20-day high, volume confirmation
            if price > hh_20 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low, volume confirmation
            elif price < ll_20 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 20-day low or 2x ATR stop
            if price < ll_20 or price < close[i-1] - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 20-day high or 2x ATR stop
            if price > hh_20 or price > close[i-1] + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_ATRStop_Session"
timeframe = "4h"
leverage = 1.0