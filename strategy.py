#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WeeklyPivot_R1S1_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using last 5 days: Monday-Friday)
    # Rolling window of 5 days for high/low/close
    high_5d = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    low_5d = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    close_5d = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot calculation
    pivot = (high_5d + low_5d + close_5d) / 3.0
    r1 = 2 * pivot - low_5d
    s1 = 2 * pivot - high_5d
    
    # Align weekly pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # Trend filter: EMA(34) on 6h close - only trade in trend direction
    ema_34 = pd.Series(close).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or \
           np.isnan(vol_ma_30[i]) or np.isnan(ema_34[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_30[i]
        
        volume_confirmed = vol > 1.8 * vol_ma
        price_above_ema = price > ema_34[i]
        price_below_ema = price < ema_34[i]
        
        if position == 0:
            # Long: Price breaks above R1 with volume and above EMA34
            if price > r1_6h[i] and volume_confirmed and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume and below EMA34
            elif price < s1_6h[i] and volume_confirmed and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below pivot OR breaks below EMA34 (trend change)
            if price < pivot_6h[i] or price < ema_34[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above pivot OR breaks above EMA34 (trend change)
            if price > pivot_6h[i] or price > ema_34[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals