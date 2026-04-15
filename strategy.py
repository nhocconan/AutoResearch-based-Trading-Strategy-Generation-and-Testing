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
    
    # Get 4h HTF data once before loop (primary timeframe alignment)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period) - HTF trend filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 6h
    upper_20_6h = align_htf_to_ltf(prices, df_4h, upper_20_4h)
    lower_20_6h = align_htf_to_ltf(prices, df_4h, lower_20_4h)
    
    # Get 1d HTF data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ATR to 6h
    atr_14_6h = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 6h ATR(14) for stoploss reference
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_6h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_14_6h_current = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Session filter: 00-24 UTC (all hours for 6h)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_6h[i]) or np.isnan(lower_20_6h[i]) or 
            np.isnan(atr_14_6h[i]) or np.isnan(atr_14_6h_current[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when 1d ATR > 0.5% of price (avoid low vol chop)
        vol_regime = atr_14_6h[i] > 0.005 * close[i]
        
        # Long conditions:
        # 1. 6h price breaks above 4h Donchian upper (20) - bullish breakout
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Volatility regime: sufficient volatility for meaningful moves
        if (close[i] > upper_20_6h[i] and
            volume_ratio[i] > 1.5 and
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 6h price breaks below 4h Donchian lower (20) - bearish breakdown
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Volatility regime: sufficient volatility for meaningful moves
        elif (close[i] < lower_20_6h[i] and
              volume_ratio[i] > 1.5 and
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_4h_Donchian20_1d_VolRegime_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0