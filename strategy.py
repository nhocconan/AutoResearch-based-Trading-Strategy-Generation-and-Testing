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
    
    # Get 1d HTF data once before loop (for 12h primary)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(21) for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_21_1d = close_1d.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Calculate 12h Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    upper_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian to 12h (no alignment needed, but using helper for consistency)
    upper_20_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_20_12h)
    lower_20_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_20_12h)
    
    # Calculate 12h ATR(14) for volatility filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    volume_ratio = vol_12h / (vol_ma_20_12h_aligned + 1e-10)
    
    signals = np.zeros(n)
    
    # Precompute session filter (8-20 UTC for institutional activity)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_12h[i]) or np.isnan(upper_20_12h_aligned[i]) or 
            np.isnan(lower_20_12h_aligned[i]) or np.isnan(atr_14_12h_aligned[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 12h price breaks above 12h Donchian upper (20) - bullish breakout
        # 2. Price above 1d EMA(21) - bullish trend filter
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        if (close[i] > upper_20_12h_aligned[i] and
            close[i] > ema_21_12h[i] and
            volume_ratio[i] > 1.5 and
            atr_14_12h_aligned[i] > 0.005 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 12h price breaks below 12h Donchian lower (20) - bearish breakdown
        # 2. Price below 1d EMA(21) - bearish trend filter
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.5% of price
        elif (close[i] < lower_20_12h_aligned[i] and
              close[i] < ema_21_12h[i] and
              volume_ratio[i] > 1.5 and
              atr_14_12h_aligned[i] > 0.005 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_12h_Donchian20_1d_EMA21_Volume_ATR_Filter_v1"
timeframe = "12h"
leverage = 1.0