#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Chandelier_Exit_Trend_Confirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Chandelier Exit and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # ATR(3) calculation for Chandelier Exit
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Chandelier Exit: 3 * ATR below highest high (long) / above lowest low (short)
    highest_high = pd.Series(high_1d).rolling(window=22, min_periods=22).max().values
    lowest_low = pd.Series(low_1d).rolling(window=22, min_periods=22).min().values
    
    long_stop = highest_high - 3.0 * atr
    short_stop = lowest_low + 3.0 * atr
    
    # Trend filter: 50 EMA on 1d
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current volume > 1.3 * 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    # Align all to 4h
    long_stop_4h = align_htf_to_ltf(prices, df_1d, long_stop)
    short_stop_4h = align_htf_to_ltf(prices, df_1d, short_stop)
    ema50_1d_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_filter_4h = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 22)
    
    for i in range(start_idx, n):
        if (np.isnan(long_stop_4h[i]) or np.isnan(short_stop_4h[i]) or
            np.isnan(ema50_1d_4h[i]) or np.isnan(volume_filter_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        long_stop_val = long_stop_4h[i]
        short_stop_val = short_stop_4h[i]
        trend = ema50_1d_4h[i]
        vol_filter = volume_filter_4h[i]
        
        if position == 0:
            # Enter long: price above long stop AND above trend AND volume confirmation
            if close[i] > long_stop_val and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: price below short stop AND below trend AND volume confirmation
            elif close[i] < short_stop_val and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price hits long stop (trailing stop)
            if close[i] <= long_stop_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price hits short stop (trailing stop)
            if close[i] >= short_stop_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals