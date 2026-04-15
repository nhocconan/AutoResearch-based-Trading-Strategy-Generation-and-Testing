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
    
    # Align 4h Donchian to 4h (no timeframe conversion needed)
    upper_20_4h = upper_20  # Already 4h resolution
    lower_20_4h = lower_20  # Already 4h resolution
    
    # Get 1d HTF data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(200) for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 4h ATR(14) for volatility filter
    # Need to calculate ATR on 4h data first, then align
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr3_4h = np.abs(low_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    # Get 4h close for ATR calculation
    close_4h = df_4h['close'].values
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    volume_ratio_4h = vol_4h / (vol_ma_20_4h + 1e-10)
    volume_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ratio_4h)
    
    signals = np.zeros(n)
    
    # Process every 4th bar (4h resolution from 1h data)
    for i in range(100, n, 4):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_4h[i//4]) or np.isnan(lower_20_4h[i//4]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(atr_14_4h_aligned[i]) or 
            np.isnan(volume_ratio_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 4h price breaks above 4h Donchian upper (20)
        # 2. 1d EMA(200) trend filter: price above EMA200 (bullish bias)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. ATR filter: ATR > 0.003 * price (avoid low volatility)
        if (close[i] > upper_20_4h[i//4] and
            close[i] > ema_200_1d_aligned[i] and
            volume_ratio_4h_aligned[i] > 1.5 and
            atr_14_4h_aligned[i] > 0.003 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 4h price breaks below 4h Donchian lower (20)
        # 2. 1d EMA(200) trend filter: price below EMA200 (bearish bias)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. ATR filter: ATR > 0.003 * price
        elif (close[i] < lower_20_4h[i//4] and
              close[i] < ema_200_1d_aligned[i] and
              volume_ratio_4h_aligned[i] > 1.5 and
              atr_14_4h_aligned[i] > 0.003 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1d_EMA200_Volume_ATR_Filter"
timeframe = "4h"
leverage = 1.0