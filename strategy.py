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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h SMA50 for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    sma50_4h = close_4h.rolling(window=50, min_periods=50).mean().values
    sma50_4h_aligned = align_htf_to_ltf(prices, df_4h, sma50_4h)
    
    # Get 1d data for Choppiness index (range detection)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of True Range over 14 periods
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(tr14) / (hh14 - ll14)) / log10(14)
    chop_raw = 100 * np.log10(atr14 / (hh14 - ll14 + 1e-10)) / np.log10(14)
    chop = pd.Series(chop_raw).fillna(50).values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume filter: require volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC (active trading hours)
    hour = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hour >= 8) & (hour <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 50  # need 50 for SMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(sma50_4h_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: CHOP > 55 (ranging) + price > SMA50 (bullish bias) + volume + session
            if (chop_aligned[i] > 55 and 
                close[i] > sma50_4h_aligned[i] and 
                volume_filter[i] and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: CHOP > 55 (ranging) + price < SMA50 (bearish bias) + volume + session
            elif (chop_aligned[i] > 55 and 
                  close[i] < sma50_4h_aligned[i] and 
                  volume_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: CHOP < 40 (trending) OR price < SMA50 (trend change)
            if (chop_aligned[i] < 40 or close[i] < sma50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: CHOP < 40 (trending) OR price > SMA50 (trend change)
            if (chop_aligned[i] < 40 or close[i] > sma50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_SMA50_1d_CHOP_Range_Volume_Session"
timeframe = "1h"
leverage = 1.0