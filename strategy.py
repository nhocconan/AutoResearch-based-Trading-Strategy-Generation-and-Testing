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
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w Donchian channels (20-period) for structure
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    upper_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align 1w Donchian to 6h
    upper_20_1w_6h = align_htf_to_ltf(prices, df_1w, upper_20_1w)
    lower_20_1w_6h = align_htf_to_ltf(prices, df_1w, lower_20_1w)
    
    # Get 1d HTF data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Pre-compute session filter (00-24 UTC for 6h - all sessions)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)  # 6h candles cover all sessions
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_1w_6h[i]) or np.isnan(lower_20_1w_6h[i]) or 
            np.isnan(ema_50_1d_6h[i]) or np.isnan(atr_14[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 6h price breaks above 1w Donchian upper (20) - bullish breakout
        # 2. 1d EMA(50) trend filter: price above EMA50 (bullish bias)
        # 3. Volume confirmation: volume > 2.0x average (strong participation)
        # 4. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        if (close[i] > upper_20_1w_6h[i] and
            close[i] > ema_50_1d_6h[i] and
            volume_ratio[i] > 2.0 and
            atr_14[i] > 0.005 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 6h price breaks below 1w Donchian lower (20) - bearish breakdown
        # 2. 1d EMA(50) trend filter: price below EMA50 (bearish bias)
        # 3. Volume confirmation: volume > 2.0x average
        # 4. Volatility filter: ATR > 0.5% of price
        elif (close[i] < lower_20_1w_6h[i] and
              close[i] < ema_50_1d_6h[i] and
              volume_ratio[i] > 2.0 and
              atr_14[i] > 0.005 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1w_Donchian20_1d_EMA50_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0