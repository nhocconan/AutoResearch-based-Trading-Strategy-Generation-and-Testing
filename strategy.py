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
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Calculate daily pivot points (Camarilla style for intraday relevance)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We'll use standard pivots but focus on R1/S1 for breakouts
    pivot = (daily_high + daily_low + daily_close) / 3.0
    r1 = pivot + (daily_high - daily_low) * 1.1 / 4  # Camarilla R3 equivalent
    s1 = pivot - (daily_high - daily_low) * 1.1 / 4  # Camarilla S3 equivalent
    r2 = pivot + (daily_high - daily_low) * 1.1 / 2  # Camarilla R4
    s2 = pivot - (daily_high - daily_low) * 1.1 / 2  # Camarilla S4
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr3 = np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe with proper delay
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    atr_14_6h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 6h Donchian channels (20-period) for breakout confirmation
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(atr_14_6h[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 6h price breaks above R1 with volume confirmation → long
        # 2. 6h price breaks below S1 with volume confirmation → short
        # 3. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        # 4. Volume confirmation: volume > 1.5x average (stricter)
        # 5. Discrete position sizing: 0.25
        # 6. Additional: price must be outside Donchian channels to avoid false breakouts
        
        # Long conditions: 6h breakout above R1
        if (close[i] > r1_6h[i] and            # 6h price above R1 pivot
            close[i] > highest_20[i] and       # Confirm with Donchian breakout
            volume_ratio[i] > 1.5 and          # Volume confirmation (stricter)
            atr_14_6h[i] > 0.005 * close[i]):  # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: 6h breakdown below S1
        elif (close[i] < s1_6h[i] and          # 6h price below S1 pivot
              close[i] < lowest_20[i] and      # Confirm with Donchian breakdown
              volume_ratio[i] > 1.5 and        # Volume confirmation (stricter)
              atr_14_6h[i] > 0.005 * close[i]): # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_Pivot_R1_S1_Breakout_Volume_Donchian_Filter"
timeframe = "6h"
leverage = 1.0