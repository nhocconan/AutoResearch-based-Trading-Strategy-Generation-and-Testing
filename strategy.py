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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_6h = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 6h Donchian channels (20-period) for breakout
    high_6h = high
    low_6h = low
    upper_20_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lower_20_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h ATR(14) for volatility filter
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.concatenate([[close[0]], close[:-1]]))
    tr3_6h = np.abs(low_6h - np.concatenate([[close[0]], close[:-1]]))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_14_6h_vol = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Precompute session filter (UTC 0-23 for 6h - all sessions)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true for 6h, kept for structure
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_6h[i]) or np.isnan(atr_14_6h[i]) or 
            np.isnan(upper_20_6h[i]) or np.isnan(lower_20_6h[i]) or 
            np.isnan(atr_14_6h_vol[i]) or np.isnan(volume_ratio[i]) or 
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: trending market when 1d ATR > 0.8% of price
        volatile_regime = atr_14_6h[i] > 0.008 * close[i]
        
        # Long conditions:
        # 1. 6h price breaks above 20-period Donchian upper
        # 2. Price above 1d EMA(34) (bullish trend bias)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatile regime filter (avoid low volatility chop)
        if (close[i] > upper_20_6h[i] and
            close[i] > ema_34_6h[i] and
            volume_ratio[i] > 1.5 and
            volatile_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 6h price breaks below 20-period Donchian lower
        # 2. Price below 1d EMA(34) (bearish trend bias)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatile regime filter
        elif (close[i] < lower_20_6h[i] and
              close[i] < ema_34_6h[i] and
              volume_ratio[i] > 1.5 and
              volatile_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1d_EMA34_ATR_Regime_Donchian20_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0