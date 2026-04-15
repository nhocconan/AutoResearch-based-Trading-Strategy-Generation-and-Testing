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
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 4h (no need to align to lower TF since we trade on 4h)
    upper_20_4h = upper_20  # Already on 4h timeframe
    lower_20_4h = lower_20  # Already on 4h timeframe
    
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
    # Get 4h close prices
    close_4h = df_4h['close'].values
    tr2_4h = np.abs(high_4h - np.concatenate([[close_4h[0]], close_4h[:-1]])) if len(close_4h) > 1 else np.zeros_like(close_4h)
    tr3_4h = np.abs(low_4h - np.concatenate([[close_4h[0]], close_4h[:-1]])) if len(close_4h) > 1 else np.zeros_like(close_4h)
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    volume_ratio_4h = vol_4h / (vol_ma_20_4h_aligned + 1e-10)
    volume_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ratio_4h)
    
    # Pre-compute session filter (08-20 UTC) for 4h bars
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Start from index where we have sufficient data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_4h[i]) or np.isnan(lower_20_4h[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_4h_aligned[i]) or 
            np.isnan(volume_ratio_4h_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 4h price breaks above 4h Donchian upper (20)
        # 2. 1d EMA(50) trend filter: price above EMA50 (bullish bias)
        # 3. Volume confirmation: volume > 2.0x average (strict)
        # 4. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        if (close[i] > upper_20_4h[i] and
            close[i] > ema_50_1d_aligned[i] and
            volume_ratio_4h_aligned[i] > 2.0 and
            atr_14_4h_aligned[i] > 0.005 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 4h price breaks below 4h Donchian lower (20)
        # 2. 1d EMA(50) trend filter: price below EMA50 (bearish bias)
        # 3. Volume confirmation: volume > 2.0x average
        # 4. Volatility filter: ATR > 0.5% of price
        elif (close[i] < lower_20_4h[i] and
              close[i] < ema_50_1d_aligned[i] and
              volume_ratio_4h_aligned[i] > 2.0 and
              atr_14_4h_aligned[i] > 0.005 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1d_EMA50_Volume_ATR_Filter"
timeframe = "4h"
leverage = 1.0