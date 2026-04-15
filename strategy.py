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
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h pivot points (standard floor trader's pivots)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    daily_high_4h = df_4h['high'].values
    daily_low_4h = df_4h['low'].values
    daily_close_4h = df_4h['close'].values
    
    pivot_4h = (daily_high_4h + daily_low_4h + daily_close_4h) / 3.0
    r1_4h = 2 * pivot_4h - daily_low_4h
    s1_4h = 2 * pivot_4h - daily_high_4h
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = pd.Series(daily_high_4h - daily_low_4h)
    tr2 = pd.Series(np.abs(daily_high_4h - np.concatenate([[daily_close_4h[0]], daily_close_4h[:-1]])))
    tr3 = pd.Series(np.abs(daily_low_4h - np.concatenate([[daily_close_4h[0]], daily_close_4h[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_4h = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 1h timeframe with proper delay
    pivot_1h = align_htf_to_ltf(prices, df_4h, pivot_4h)
    r1_1h = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_1h = align_htf_to_ltf(prices, df_4h, s1_4h)
    atr_14_1h = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Calculate 1h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1h[i]) or np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or 
            np.isnan(atr_14_1h[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 1h price breaks above R1 with volume confirmation → long
        # 2. 1h price breaks below S1 with volume confirmation → short
        # 3. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        # 4. Volume confirmation: volume > 1.3x average
        # 5. Discrete position sizing: 0.20
        
        # Long conditions: 1h breakout above R1
        if (close[i] > r1_1h[i] and            # 1h price above R1 pivot
            volume_ratio[i] > 1.3 and          # Volume confirmation
            atr_14_1h[i] > 0.005 * close[i]):  # Volatility filter
            signals[i] = 0.20
            
        # Short conditions: 1h breakdown below S1
        elif (close[i] < s1_1h[i] and          # 1h price below S1 pivot
              volume_ratio[i] > 1.3 and        # Volume confirmation
              atr_14_1h[i] > 0.005 * close[i]): # Volatility filter
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Pivot_R1_S1_Breakout_Volume_ATR_Filter"
timeframe = "1h"
leverage = 1.0