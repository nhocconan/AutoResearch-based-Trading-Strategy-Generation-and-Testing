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
    
    # Get daily data for indicator calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR for volatility
    tr_1d = np.maximum(
        high_1d - low_1d,
        np.maximum(
            np.abs(high_1d - np.roll(close_1d, 1)),
            np.abs(low_1d - np.roll(close_1d, 1))
        )
    )
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = np.zeros_like(tr_1d)
    for i in range(len(tr_1d)):
        if i < 14:
            atr_1d[i] = np.mean(tr_1d[:i+1]) if i > 0 else tr_1d[i]
        else:
            atr_1d[i] = 0.93 * atr_1d[i-1] + 0.07 * tr_1d[i]
    
    # Calculate daily moving averages
    ma_5_1d = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    ma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily volume average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    ma_5_1d_aligned = align_htf_to_ltf(prices, df_1d, ma_5_1d)
    ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ma_20_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ma_5_1d_aligned[i]) or 
            np.isnan(ma_20_1d_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility (dead markets)
        volatility_filter = atr_1d_aligned[i] > 0.0001 * close[i]  # At least 0.01% ATR
        
        # Volume filter: above average volume
        volume_filter = volume[i] > vol_ma_20_1d_aligned[i]
        
        # Trend filters
        bullish = close[i] > ma_5_1d_aligned[i] and ma_5_1d_aligned[i] > ma_20_1d_aligned[i]
        bearish = close[i] < ma_5_1d_aligned[i] and ma_5_1d_aligned[i] < ma_20_1d_aligned[i]
        
        # Entry conditions
        long_entry = bullish and volatility_filter and volume_filter
        short_entry = bearish and volatility_filter and volume_filter
        
        # Exit conditions: trend reversal
        exit_long = position == 1 and not bullish
        exit_short = position == -1 and not bearish
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_ma5_ma20_volume_filter_v1"
timeframe = "4h"
leverage = 1.0