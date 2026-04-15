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
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 4h timeframe (no additional alignment needed as we trade on 4h)
    upper_20_4h = upper_20
    lower_20_4h = lower_20
    
    # Get 1d HTF data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter (shorter for better responsiveness)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h ATR(14) for volatility filter
    # Need to calculate TR using 4h data
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.concatenate([[close_4h[0]], close_4h[:-1]])) if len(close_4h) > 1 else np.array([0])
    tr3_4h = np.abs(low_4h - np.concatenate([[close_4h[0]], close_4h[:-1]])) if len(close_4h) > 1 else np.array([0])
    # Get 4h close for TR calculation
    close_4h = df_4h['close'].values
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr3_4h = np.abs(low_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    volume_ratio_4h = vol_4h / (vol_ma_20_4h_aligned + 1e-10)
    
    # Pre-compute session filter (08-20 UTC) - using 4h bar times
    # For 4h bars, we check if the bar's timestamp hour is in session
    hours_4h = df_4h.index.hour.values
    in_session_4h = (hours_4h >= 8) & (hours_4h <= 20)
    in_session_aligned = align_htf_to_ltf(prices, df_4h, in_session_4h.astype(float))
    
    signals = np.zeros(n)
    
    # Start loop after sufficient warmup for indicators
    start_idx = max(100, 50)  # Ensure we have enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_4h[i]) or np.isnan(lower_20_4h[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_4h_aligned[i]) or 
            np.isnan(volume_ratio_4h[i]) or np.isnan(in_session_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 4h price breaks above 4h Donchian upper (20)
        # 2. 1d EMA(50) trend filter: price above EMA50 (bullish bias)
        # 3. Volume confirmation: volume > 1.8x average (higher threshold for fewer trades)
        # 4. Session filter: trade only during active hours
        if (close[i] > upper_20_4h[i] and
            close[i] > ema_50_1d_aligned[i] and
            volume_ratio_4h[i] > 1.8 and
            in_session_aligned[i] > 0.5):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 4h price breaks below 4h Donchian lower (20)
        # 2. 1d EMA(50) trend filter: price below EMA50 (bearish bias)
        # 3. Volume confirmation: volume > 1.8x average
        # 4. Session filter: trade only during active hours
        elif (close[i] < lower_20_4h[i] and
              close[i] < ema_50_1d_aligned[i] and
              volume_ratio_4h[i] > 1.8 and
              in_session_aligned[i] > 0.5):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1d_EMA50_Volume_Session_Filter"
timeframe = "4h"
leverage = 1.0