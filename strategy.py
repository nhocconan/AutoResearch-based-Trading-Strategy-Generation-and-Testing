#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA(21) for trend filter
    ema_21_12h = pd.Series(df_12h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Calculate 12h ATR(14) for volatility filter
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = np.abs(df_12h['high'] - np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]]))
    tr3 = np.abs(df_12h['low'] - np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]]))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # Calculate 12h Donchian(20) channels
    donchian_high_20 = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_20)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(ema_21_12h_aligned[i]) or np.isnan(atr_14_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when 12h ATR is elevated (> 0.4% of price)
        vol_filter = atr_14_12h_aligned[i] > 0.004 * close[i]
        
        # Long conditions:
        # 1. Price above 12h EMA21 (bullish bias)
        # 2. Price breaks above 12h Donchian(20) high
        # 3. Volatility filter
        if (close[i] > ema_21_12h_aligned[i] and
            close[i] > donchian_high_20_aligned[i] and
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below 12h EMA21 (bearish bias)
        # 2. Price breaks below 12h Donchian(20) low
        # 3. Volatility filter
        elif (close[i] < ema_21_12h_aligned[i] and
              close[i] < donchian_low_20_aligned[i] and
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_EMA21_Donchian20_VolFilter_v1"
timeframe = "4h"
leverage = 1.0