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
    
    # Get 1d HTF data once before loop for weekly pivot levels
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
    
    # Align weekly pivot levels to 4h
    weekly_pivot_4h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_4h = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_4h = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
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
    
    # Precompute session filter (00-24 UTC for 4h - less restrictive)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true for 4h, kept for structure
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_4h[i]) or np.isnan(weekly_r1_4h[i]) or 
            np.isnan(weekly_s1_4h[i]) or np.isnan(atr_14[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 4h price breaks above weekly R1 - bullish breakout above resistance
        # 2. Volume confirmation: volume > 1.3x average
        # 3. Volatility filter: ATR > 0.4% of price (avoid low volatility chop)
        if (close[i] > weekly_r1_4h[i] and
            volume_ratio[i] > 1.3 and
            atr_14[i] > 0.004 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 4h price breaks below weekly S1 - bearish breakdown below support
        # 2. Volume confirmation: volume > 1.3x average
        # 3. Volatility filter: ATR > 0.4% of price
        elif (close[i] < weekly_s1_4h[i] and
              volume_ratio[i] > 1.3 and
              atr_14[i] > 0.004 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_WeeklyPivot_R1S1_Breakout_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0