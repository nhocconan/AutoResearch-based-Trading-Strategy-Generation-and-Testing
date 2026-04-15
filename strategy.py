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
    
    # Calculate 4h ATR(14) for volatility filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr3 = np.abs(low_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 4h ATR to 4h (no conversion needed as we're staying in 4h)
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Get 1d HTF data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h Donchian channels (20-period)
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 4h
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_4h = df_4h['volume'].values
    vol_ma_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    volume_ratio = vol_4h / (vol_ma_20_aligned + 1e-10)
    
    signals = np.zeros(n)
    
    # Pre-compute session filter (08-20 UTC) for 4h bars
    # For 4h timeframe, we check if the bar's opening hour is in session
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_4h_aligned[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 4h price breaks above 4h Donchian upper (20)
        # 2. 1d EMA(50) trend filter: price above EMA50 (bullish bias)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. ATR filter: current ATR > 50% of 4h ATR (avoid extremely low volatility)
        if (close[i] > upper_20_aligned[i] and
            close[i] > ema_50_1d_aligned[i] and
            volume_ratio[i] > 1.5 and
            atr_14_4h_aligned[i] > 0.5 * atr_14_4h[i]):  # Using 4h ATR for consistency
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 4h price breaks below 4h Donchian lower (20)
        # 2. 1d EMA(50) trend filter: price below EMA50 (bearish bias)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. ATR filter: current ATR > 50% of 4h ATR
        elif (close[i] < lower_20_aligned[i] and
              close[i] < ema_50_1d_aligned[i] and
              volume_ratio[i] > 1.5 and
              atr_14_4h_aligned[i] > 0.5 * atr_14_4h[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1d_EMA50_Volume_ATR_Filter_v1"
timeframe = "4h"
leverage = 1.0