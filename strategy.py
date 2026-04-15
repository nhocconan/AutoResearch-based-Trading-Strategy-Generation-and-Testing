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
    
    # Get weekly HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily Donchian(20) for breakout signals
    donchian_high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when daily ATR is elevated (> 0.4% of price)
        vol_filter = atr_14_1d_aligned[i] > 0.004 * close[i]
        
        # Long conditions:
        # 1. Price above weekly EMA34 (bullish bias from higher timeframe)
        # 2. Price breaks above daily Donchian(20) high
        # 3. Volatility filter
        if (close[i] > ema_34_1w_aligned[i] and
            close[i] > donchian_high_20_aligned[i] and
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below weekly EMA34 (bearish bias from higher timeframe)
        # 2. Price breaks below daily Donchian(20) low
        # 3. Volatility filter
        elif (close[i] < ema_34_1w_aligned[i] and
              close[i] < donchian_low_20_aligned[i] and
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_EMA34_1w_Donchian20_VolFilter_v1"
timeframe = "1d"
leverage = 1.0