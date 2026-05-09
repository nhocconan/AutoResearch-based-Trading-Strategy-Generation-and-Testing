#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_ChoppinessIndex_MeanReversion_V2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for choppiness and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (14-period)
    atr_series = []
    for i in range(len(df_1d)):
        if i == 0:
            tr = df_1d['high'].iloc[i] - df_1d['low'].iloc[i]
        else:
            tr = max(
                df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
                abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
                abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
            )
        atr_series.append(tr)
    
    atr_series = np.array(atr_series)
    atr_sum = pd.Series(atr_series).rolling(window=14, min_periods=14).sum().values
    
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_sum = highest_high - lowest_low
    chop = np.zeros_like(atr_sum)
    mask = range_sum != 0
    chop[mask] = 100 * np.log10(atr_sum[mask] / range_sum[mask]) / np.log10(14)
    
    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current 1d volume > 1.3 * 30-day average
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    volume_filter_1d = df_1d['volume'].values > (vol_ma * 1.3)
    
    # Align all to 12h
    chop_12h = align_htf_to_ltf(prices, df_1d, chop)
    ema50_1d_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_filter_12h = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 30)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(chop_12h[i]) or np.isnan(ema50_1d_12h[i]) or 
            np.isnan(volume_filter_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_12h[i]
        trend = ema50_1d_12h[i]
        vol_filter = volume_filter_12h[i]
        
        if position == 0:
            # Enter long in ranging market (high chop) when price below EMA50
            if chop_val > 61.8 and close[i] < trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short in ranging market (high chop) when price above EMA50
            elif chop_val > 61.8 and close[i] > trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: chop drops (trending) or price crosses above EMA50
            if chop_val < 38.2 or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: chop drops (trending) or price crosses below EMA50
            if chop_val < 38.2 or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals