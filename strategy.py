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
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d ADX(14) for trend strength filter
    plus_dm = np.where((df_1d['high'] - df_1d['high'].shift(1)) > (df_1d['low'].shift(1) - df_1d['low']), 
                       np.maximum(df_1d['high'] - df_1d['high'].shift(1), 0), 0)
    minus_dm = np.where((df_1d['low'].shift(1) - df_1d['low']) > (df_1d['high'] - df_1d['high'].shift(1)), 
                        np.maximum(df_1d['low'].shift(1) - df_1d['low'], 0), 0)
    tr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean()
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean() / tr_14
    dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14_1d = pd.Series(dx_14).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 4h
    atr_14_4h = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    adx_14_4h = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Calculate 4h Donchian channels (20-period)
    upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR(14) for stoploss
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_4h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_14_4h_local = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(atr_14_4h[i]) or np.isnan(adx_14_4h[i]) or 
            np.isnan(atr_14_4h_local[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 4h price breaks above 4h Donchian upper (20) - bullish breakout
        # 2. 1d ATR > 1.5% of price (sufficient volatility)
        # 3. 1d ADX > 25 (trending market)
        if (close[i] > upper_20[i] and
            atr_14_4h[i] > 0.015 * close[i] and
            adx_14_4h[i] > 25):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 4h price breaks below 4h Donchian lower (20) - bearish breakdown
        # 2. 1d ATR > 1.5% of price (sufficient volatility)
        # 3. 1d ADX > 25 (trending market)
        elif (close[i] < lower_20[i] and
              atr_14_4h[i] > 0.015 * close[i] and
              adx_14_4h[i] > 25):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1d_ATR_ADX_Filter_v1"
timeframe = "4h"
leverage = 1.0