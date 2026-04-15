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
    
    # Calculate 4h ATR(14) for volatility and Donchian channels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range for 4h
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr3_4h = np.abs(low_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 4h Donchian channels (20-period)
    upper_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h indicators to 1h timeframe
    upper_20_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_20_4h)
    lower_20_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_20_4h)
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Get 1d HTF data for trend filter (EMA 50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_4h_aligned[i]) or np.isnan(lower_20_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_4h_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 1h price breaks above 4h Donchian upper (20) with buffer
        # 2. 1d EMA(50) trend filter: price above EMA50 (bullish bias)
        # 3. Volume confirmation: volume > 2.0x average (strict)
        # 4. Volatility filter: ATR > 0.5% of price AND 4h ATR > 0.3% of 4h price
        if (close[i] > upper_20_4h_aligned[i] * 1.001 and
            close[i] > ema_50_1d_aligned[i] and
            volume_ratio[i] > 2.0 and
            atr_14[i] > 0.005 * close[i] and
            atr_14_4h_aligned[i] > 0.003 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 1h price breaks below 4h Donchian lower (20) with buffer
        # 2. 1d EMA(50) trend filter: price below EMA50 (bearish bias)
        # 3. Volume confirmation: volume > 2.0x average
        # 4. Volatility filter: ATR > 0.5% of price AND 4h ATR > 0.3% of 4h price
        elif (close[i] < lower_20_4h_aligned[i] * 0.999 and
              close[i] < ema_50_1d_aligned[i] and
              volume_ratio[i] > 2.0 and
              atr_14[i] > 0.005 * close[i] and
              atr_14_4h_aligned[i] > 0.003 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4h_Donchian20_1d_EMA50_Volume_ATR_Filter_v1"
timeframe = "1h"
leverage = 1.0