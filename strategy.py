#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop (for weekly pivot levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week (using 1d data)
    # Weekly high/low/close from 5 trading days ago (prior week)
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(5).values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(5).values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(5).values
    
    # Weekly pivot: (H+L+C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly R1: 2*P - L
    weekly_r1 = 2 * weekly_pivot - weekly_low
    # Weekly S1: 2*P - H
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivot levels to 12h
    weekly_pivot_12h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_12h = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_12h = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ATR(14) for volatility filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 12h ATR to 12h (no shift needed as it's already HTF)
    atr_14_12h_aligned = atr_14_12h  # Already at 12h frequency
    
    # Calculate 12h volume ratio (current vs 20-period average)
    volume_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ratio_12h = volume_12h / (vol_ma_20_12h + 1e-10)
    
    # Align 12h volume ratio to 12h
    volume_ratio_12h_aligned = volume_ratio_12h  # Already at 12h frequency
    
    signals = np.zeros(n)
    
    # Precompute session filter (8-20 UTC for 12h - avoids Asian session lows)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_12h[i]) or np.isnan(weekly_r1_12h[i]) or 
            np.isnan(weekly_s1_12h[i]) or np.isnan(atr_14_12h_aligned[i]) or 
            np.isnan(volume_ratio_12h_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Get current 12h price (use close of completed 12h bar)
        # Since we're on 12h timeframe, close[i] is the close of the 12h bar at index i
        price = close[i]
        
        # Long conditions:
        # 1. Price breaks above weekly R1 (bullish breakout from prior week resistance)
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        if (price > weekly_r1_12h[i] and
            volume_ratio_12h_aligned[i] > 1.5 and
            atr_14_12h_aligned[i] > 0.005 * price):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below weekly S1 (bearish breakdown from prior week support)
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Volatility filter: ATR > 0.5% of price
        elif (price < weekly_s1_12h[i] and
              volume_ratio_12h_aligned[i] > 1.5 and
              atr_14_12h_aligned[i] > 0.005 * price):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_1d_WeeklyPivot_R1S1_Breakout_Volume_Volatility_Filter_v1"
timeframe = "12h"
leverage = 1.0