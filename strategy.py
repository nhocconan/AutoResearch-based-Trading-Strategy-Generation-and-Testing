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
    
    # Get weekly data for trend filter (use 200-bar EMA on weekly closes)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    weekly_ema200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    weekly_ema200_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema200)
    
    # Get daily data for 12-period ATR and volume average
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12-period ATR (using true range)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])  # align with index
    atr_12 = pd.Series(tr).rolling(window=12, min_periods=12).mean().values
    atr_12_aligned = align_htf_to_ltf(prices, df_1d, atr_12)
    
    # Calculate daily volume average (20-period)
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_ema200_aligned[i]) or np.isnan(atr_12_aligned[i]) or
            np.isnan(volume_ma20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-day average
        volume_filter = volume[i] > (1.5 * volume_ma20_1d_aligned[i])
        
        # Trend filter: price relative to weekly EMA200
        price_above_weekly_ema = close[i] > weekly_ema200_aligned[i]
        price_below_weekly_ema = close[i] < weekly_ema200_aligned[i]
        
        # ATR-based entry threshold
        atr_threshold = 0.5 * atr_12_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above weekly EMA200 + ATR threshold with volume
            if (close[i] > weekly_ema200_aligned[i] + atr_threshold and 
                volume_filter and price_above_weekly_ema):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly EMA200 - ATR threshold with volume
            elif (close[i] < weekly_ema200_aligned[i] - atr_threshold and 
                  volume_filter and price_below_weekly_ema):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly EMA200
            if close[i] < weekly_ema200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly EMA200
            if close[i] > weekly_ema200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyEMA200_ATR_Breakout_Volume"
timeframe = "12h"
leverage = 1.0