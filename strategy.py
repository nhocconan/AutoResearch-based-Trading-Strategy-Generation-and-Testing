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
    
    # Get 1d HTF data once before loop (for 12h primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily Donchian channels (20-period) for breakout signals
    upper_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 12h timeframe
    atr_14_12h = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    upper_20_12h = align_htf_to_ltf(prices, df_1d, upper_20_1d)
    lower_20_12h = align_htf_to_ltf(prices, df_1d, lower_20_1d)
    
    # Calculate 12h ATR(14) for stoploss and volatility filter
    tr1_12h = high - low
    tr2_12h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_12h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_14_12h_internal = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Session filter: 00-24 UTC (always true for 12h, kept for structure)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_12h[i]) or np.isnan(upper_20_12h[i]) or 
            np.isnan(lower_20_12h[i]) or np.isnan(atr_14_12h_internal[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 12h price breaks above 1d Donchian upper (20) - bullish breakout
        # 2. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        # 3. Volume confirmation: volume > 1.5x average
        if (close[i] > upper_20_12h[i] and
            atr_14_12h_internal[i] > 0.005 * close[i] and
            volume_ratio[i] > 1.5):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 12h price breaks below 1d Donchian lower (20) - bearish breakdown
        # 2. Volatility filter: ATR > 0.5% of price
        # 3. Volume confirmation: volume > 1.5x average
        elif (close[i] < lower_20_12h[i] and
              atr_14_12h_internal[i] > 0.005 * close[i] and
              volume_ratio[i] > 1.5):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_1d_Donchian20_Volume_ATR_Filter_v1"
timeframe = "12h"
leverage = 1.0