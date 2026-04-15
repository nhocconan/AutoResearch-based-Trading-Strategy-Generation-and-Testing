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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian to 4h
    upper_20_4h = align_htf_to_ltf(prices, df_1d, upper_20_1d)
    lower_20_4h = align_htf_to_ltf(prices, df_1d, lower_20_1d)
    
    # Get 1w HTF data for weekly pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week (using 1w data)
    # Weekly high/low/close from prior week
    weekly_high = pd.Series(df_1w['high']).rolling(window=1, min_periods=1).max().shift(1).values
    weekly_low = pd.Series(df_1w['low']).rolling(window=1, min_periods=1).min().shift(1).values
    weekly_close = pd.Series(df_1w['close']).rolling(window=1, min_periods=1).last().shift(1).values
    
    # Weekly pivot: (H+L+C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly R1: 2*P - L
    weekly_r1 = 2 * weekly_pivot - weekly_low
    # Weekly S1: 2*P - H
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivot levels to 4h
    weekly_pivot_4h = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_4h = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_4h = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Calculate 4h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_4h[i]) or np.isnan(lower_20_4h[i]) or 
            np.isnan(weekly_pivot_4h[i]) or np.isnan(weekly_r1_4h[i]) or 
            np.isnan(weekly_s1_4h[i]) or np.isnan(atr_14[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 4h price breaks above 1d Donchian upper (20) - bullish breakout
        # 2. Price above weekly pivot (bullish bias from prior week)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        if (close[i] > upper_20_4h[i] and
            close[i] > weekly_pivot_4h[i] and
            volume_ratio[i] > 1.5 and
            atr_14[i] > 0.005 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 4h price breaks below 1d Donchian lower (20) - bearish breakdown
        # 2. Price below weekly pivot (bearish bias from prior week)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.5% of price
        elif (close[i] < lower_20_4h[i] and
              close[i] < weekly_pivot_4h[i] and
              volume_ratio[i] > 1.5 and
              atr_14[i] > 0.005 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_Donchian20_1w_WeeklyPivot_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0