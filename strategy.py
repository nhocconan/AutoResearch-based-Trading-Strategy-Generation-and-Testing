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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Supertrend (ATR=10, mult=3.0) - proven BTC/ETH edge
    hl2 = (df_12h['high'] + df_12h['low']) / 2
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = np.abs(df_12h['high'] - np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]]))
    tr3 = np.abs(df_12h['low'] - np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]]))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr_12h).ewm(span=10, adjust=False, min_periods=10).mean().values
    upper_band = hl2 + 3.0 * atr_12h
    lower_band = hl2 - 3.0 * atr_12h
    
    # Initialize Supertrend
    supertrend = np.full_like(close, np.nan, dtype=float)
    direction = np.full_like(close, np.nan, dtype=float)  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(df_12h)):
        if i == 0:
            supertrend[i] = upper_band[i]
            direction[i] = 1
        else:
            if close_12h := df_12h['close'].iloc[i]:
                if close_12h > upper_band[i-1]:
                    direction[i] = 1
                elif close_12h < lower_band[i-1]:
                    direction[i] = -1
                else:
                    direction[i] = direction[i-1]
                
                if direction[i] == 1:
                    supertrend[i] = max(lower_band[i], supertrend[i-1])
                else:
                    supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Align 12h Supertrend to 6h
    supertrend_6h = align_htf_to_ltf(prices, df_12h, supertrend)
    direction_6h = align_htf_to_ltf(prices, df_12h, direction)
    
    # Get 1d HTF data for weekly pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week (using 1d data)
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
    
    # Precompute session filter (00-24 UTC for 6h - less restrictive)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true for 6h, kept for structure
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_6h[i]) or np.isnan(direction_6h[i]) or 
            np.isnan(weekly_pivot_6h[i]) or np.isnan(weekly_r1_6h[i]) or 
            np.isnan(weekly_s1_6h[i]) or np.isnan(atr_14[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 12h Supertrend uptrend (direction = 1)
        # 2. 6h price above weekly pivot (bullish bias from prior week)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        if (direction_6h[i] == 1 and
            close[i] > weekly_pivot_6h[i] and
            volume_ratio[i] > 1.5 and
            atr_14[i] > 0.005 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 12h Supertrend downtrend (direction = -1)
        # 2. 6h price below weekly pivot (bearish bias from prior week)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.5% of price
        elif (direction_6h[i] == -1 and
              close[i] < weekly_pivot_6h[i] and
              volume_ratio[i] > 1.5 and
              atr_14[i] > 0.005 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_12h_Supertrend_1d_WeeklyPivot_Volume_ATR_Filter_v1"
timeframe = "6h"
leverage = 1.0