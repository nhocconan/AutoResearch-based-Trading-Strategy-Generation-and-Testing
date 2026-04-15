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
    
    # Calculate 4h Donchian channels (25-period for fewer signals)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper_25 = pd.Series(high_4h).rolling(window=25, min_periods=25).max().values
    lower_25 = pd.Series(low_4h).rolling(window=25, min_periods=25).min().values
    
    # Align 4h Donchian to 1h
    upper_25_1h = align_htf_to_ltf(prices, df_4h, upper_25)
    lower_25_1h = align_htf_to_ltf(prices, df_4h, lower_25)
    
    # Get 1d HTF data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter (faster than 200)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1h ATR(10) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 1h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_25_1h[i]) or np.isnan(lower_25_1h[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_10[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 1h price breaks above 4h Donchian upper (25)
        # 2. 1d EMA(50) trend filter: price above EMA50 (bullish bias)
        # 3. Volume confirmation: volume > 1.3x average (slightly reduced)
        # 4. Volatility filter: ATR > 0.25% of price (reduced threshold)
        if (close[i] > upper_25_1h[i] and
            close[i] > ema_50_1d_aligned[i] and
            volume_ratio[i] > 1.3 and
            atr_10[i] > 0.0025 * close[i]):
            signals[i] = 0.20
            
        # Short conditions:
        # 1. 1h price breaks below 4h Donchian lower (25)
        # 2. 1d EMA(50) trend filter: price below EMA50 (bearish bias)
        # 3. Volume confirmation: volume > 1.3x average
        # 4. Volatility filter: ATR > 0.25% of price
        elif (close[i] < lower_25_1h[i] and
              close[i] < ema_50_1d_aligned[i] and
              volume_ratio[i] > 1.3 and
              atr_10[i] > 0.0025 * close[i]):
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4h_Donchian25_1d_EMA50_Volume_Filter_v3"
timeframe = "1h"
leverage = 1.0