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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d ADX(14) for trend strength
    plus_dm = np.where((df_1d['high'] - df_1d['high'].shift(1)) > (df_1d['low'].shift(1) - df_1d['low']), 
                       np.maximum(df_1d['high'] - df_1d['high'].shift(1), 0), 0)
    minus_dm = np.where((df_1d['low'].shift(1) - df_1d['low']) > (df_1d['high'] - df_1d['high'].shift(1)), 
                        np.maximum(df_1d['low'].shift(1) - df_1d['low'], 0), 0)
    tr_1d_series = pd.Series(tr_1d)
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / tr_1d_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / tr_1d_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = pd.Series(dx_14).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 6h
    atr_14_6h = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    adx_14_6h = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 6h Donchian channels (20-period)
    upper_20_6h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_20_6h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Precompute session filter (00-24 UTC for 6h - less restrictive)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true for 6h, kept for structure
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_6h[i]) or np.isnan(lower_20_6h[i]) or 
            np.isnan(atr_14_6h[i]) or np.isnan(adx_14_6h[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 6h price breaks above 6h Donchian upper (20) - bullish breakout
        # 2. Strong trend: ADX > 25
        # 3. Low volatility regime: ATR < 0.8% of price (avoid high volatility chop)
        # 4. Volume confirmation: volume > 1.2x average
        if (close[i] > upper_20_6h[i] and
            adx_14_6h[i] > 25 and
            atr_14_6h[i] < 0.008 * close[i] and
            volume_ratio[i] > 1.2):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 6h price breaks below 6h Donchian lower (20) - bearish breakdown
        # 2. Strong trend: ADX > 25
        # 3. Low volatility regime: ATR < 0.8% of price
        # 4. Volume confirmation: volume > 1.2x average
        elif (close[i] < lower_20_6h[i] and
              adx_14_6h[i] > 25 and
              atr_14_6h[i] < 0.008 * close[i] and
              volume_ratio[i] > 1.2):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_ADX25_Donchian20_LowVol_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0