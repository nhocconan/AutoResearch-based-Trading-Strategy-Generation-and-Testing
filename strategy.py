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
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA(21) for trend filter
    ema_21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1h Donchian(20) breakout levels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Session filter: 08-20 UTC (precompute for efficiency)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            signals[i] = 0.0
            continue
        
        # Check session filter
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when daily ATR is elevated (> 0.5% of price)
        vol_regime = atr_14_1d_aligned[i] > 0.005 * close[i]
        
        # Long conditions:
        # 1. 4h EMA(21) uptrend (price above EMA)
        # 2. Price breaks above 1h Donchian(20) high with volume confirmation
        # 3. Daily volatility regime filter
        if (close[i] > ema_21_4h_aligned[i] and
            close[i] > highest_20[i] and
            volume[i] > 1.5 * np.mean(volume[max(0, i-20):i]) and
            vol_regime):
            signals[i] = 0.20
            
        # Short conditions:
        # 1. 4h EMA(21) downtrend (price below EMA)
        # 2. Price breaks below 1h Donchian(20) low with volume confirmation
        # 3. Daily volatility regime filter
        elif (close[i] < ema_21_4h_aligned[i] and
              close[i] < lowest_20[i] and
              volume[i] > 1.5 * np.mean(volume[max(0, i-20):i]) and
              vol_regime):
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_EMA21_4h_Donchian20_Volume_Breakout_v1"
timeframe = "1h"
leverage = 1.0