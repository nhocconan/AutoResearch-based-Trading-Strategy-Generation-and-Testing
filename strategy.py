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
    
    # Get 1d HTF data once before loop (for Donchian and ATR)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian(20) channels
    donchian_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (no shift needed as we use prior day's levels)
    donchian_high_1d = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_1d = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = df_1d['high'].values - df_1d['low'].values
    tr2 = np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))
    tr3 = np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Get 1w HTF data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_1d[i]) or np.isnan(donchian_low_1d[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr_14_1d[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Weekly trend filter: price above 1w EMA20 (bullish weekly bias)
        # 2. Price breaks above prior day's Donchian high (breakout)
        # 3. Volatility filter: ATR > 0.5% of price (avoid extremely low volatility)
        if (close[i] > ema_20_1w_aligned[i] and
            close[i] > donchian_high_1d[i] and
            atr_14_1d[i] > 0.005 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Weekly trend filter: price below 1w EMA20 (bearish weekly bias)
        # 2. Price breaks below prior day's Donchian low (breakdown)
        # 3. Volatility filter: ATR > 0.5% of price
        elif (close[i] < ema_20_1w_aligned[i] and
              close[i] < donchian_low_1d[i] and
              atr_14_1d[i] > 0.005 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_1w_EMA20_Donchian20_Breakout_Vol_Filter_v1"
timeframe = "1d"
leverage = 1.0