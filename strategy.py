#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend context and signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily SMA(50) for trend
    close_1d_series = pd.Series(close_1d)
    sma_50_1d = close_1d_series.rolling(window=50, min_periods=50).mean().values
    
    # Calculate daily ATR(14)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_1d[i] = np.mean(tr[i-14:i+1])
    
    # Calculate daily Donchian(10) channels
    high_10_1d = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    low_10_1d = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    
    # Calculate daily volume moving average
    vol_s_1d = pd.Series(volume_1d)
    vol_ma_20_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 12h timeframe
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    high_10_1d_aligned = align_htf_to_ltf(prices, df_1d, high_10_1d)
    low_10_1d_aligned = align_htf_to_ltf(prices, df_1d, low_10_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(sma_50_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(high_10_1d_aligned[i]) or np.isnan(low_10_1d_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-period daily volume MA
        vol_filter = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        # Trend filter: price above/below daily SMA50
        uptrend = close[i] > sma_50_1d_aligned[i]
        downtrend = close[i] < sma_50_1d_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > high_10_1d_aligned[i]
        short_breakout = close[i] < low_10_1d_aligned[i]
        
        # Entry conditions: breakout in trend direction + volume filter
        long_entry = long_breakout and uptrend and vol_filter
        short_entry = short_breakout and downtrend and vol_filter
        
        # Exit conditions: opposite breakout
        long_exit = close[i] < low_10_1d_aligned[i]
        short_exit = close[i] > high_10_1d_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_sma50_donchian10_vol_filter_v1"
timeframe = "12h"
leverage = 1.0