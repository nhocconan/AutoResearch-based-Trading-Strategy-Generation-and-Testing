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
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA(21) for trend filter
    ema_21_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Calculate 1h Donchian(20) for entry timing
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(donchian_high_20[i]) or 
            np.isnan(donchian_low_20[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when 1h ATR is elevated (> 0.5% of price)
        vol_filter = atr_14[i] > 0.005 * close[i]
        
        # Long conditions:
        # 1. 4h EMA21 uptrend (price above EMA21)
        # 2. Price breaks above 1h Donchian(20) high
        # 3. Volatility filter
        if (close[i] > ema_21_4h_aligned[i] and
            close[i] > donchian_high_20[i] and
            vol_filter):
            signals[i] = 0.20
            
        # Short conditions:
        # 1. 4h EMA21 downtrend (price below EMA21)
        # 2. Price breaks below 1h Donchian(20) low
        # 3. Volatility filter
        elif (close[i] < ema_21_4h_aligned[i] and
              close[i] < donchian_low_20[i] and
              vol_filter):
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_EMA21_4h_Donchian20_VolFilter_v1"
timeframe = "1h"
leverage = 1.0