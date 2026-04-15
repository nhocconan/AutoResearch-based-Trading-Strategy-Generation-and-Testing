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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d weekly pivot points (from prior week)
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
    
    # Align weekly pivot levels to 6h
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_6h = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Precompute session filter (00-24 UTC for 6h - always true, kept for structure)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_6h[i]) or np.isnan(weekly_r1_6h[i]) or 
            np.isnan(weekly_s1_6h[i]) or np.isnan(atr_14[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 6h price breaks above weekly R1 (bullish breakout)
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        if (close[i] > weekly_r1_6h[i] and
            volume_ratio[i] > 1.5 and
            atr_14[i] > 0.005 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 6h price breaks below weekly S1 (bearish breakdown)
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Volatility filter: ATR > 0.5% of price
        elif (close[i] < weekly_s1_6h[i] and
              volume_ratio[i] > 1.5 and
              atr_14[i] > 0.005 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1d_WeeklyPivot_R1S1_Breakout_Volume_ATR_Filter_v1"
timeframe = "6h"
leverage = 1.0