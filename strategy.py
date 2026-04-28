#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ATR(14) for volatility filter
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Get 6h data for Donchian channel and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # 6h Donchian(20) breakout levels
    highest_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    highest_20_aligned = align_htf_to_ltf(prices, df_6h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_6h, lowest_20)
    
    # 6h volume ratio (current volume / 20-period average)
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(highest_20_aligned[i]) or
            np.isnan(lowest_20_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volatility filter: ATR > 0.5% of price
        vol_filter = atr_1d_aligned[i] > 0.005 * close[i]
        
        # Volume filter: current 6h volume above average
        volume_filter = volume_6h[i] > vol_ma_20_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > highest_20_aligned[i]
        short_breakout = close[i] < lowest_20_aligned[i]
        
        # Entry conditions: Donchian breakout with 1d trend, volume, and volatility
        long_entry = uptrend and long_breakout and volume_filter and vol_filter
        short_entry = downtrend and short_breakout and volume_filter and vol_filter
        
        # Exit conditions: Donchian midpoint retracement
        midpoint = (highest_20_aligned[i] + lowest_20_aligned[i]) / 2
        long_exit = close[i] < midpoint
        short_exit = close[i] > midpoint
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_1dTrend_VolumeVolatilityFilter"
timeframe = "6h"
leverage = 1.0