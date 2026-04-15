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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Vectorized rolling max/min for Donchian
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe (wait for completed 12h bar)
    dh_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    dl_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Calculate 12h ATR(14) for volatility filter
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[high_12h[0]], high_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[low_12h[0]], low_12h[:-1]]))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # Calculate 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(dh_aligned[i]) or np.isnan(dl_aligned[i]) or 
            np.isnan(atr_14_12h_aligned[i]) or np.isnan(ema_34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when 12h ATR is elevated (> 0.5% of price)
        vol_filter = atr_14_12h_aligned[i] > 0.005 * close[i]
        
        # Long conditions:
        # 1. Price breaks above 12h Donchian high (breakout)
        # 2. Price above 12h EMA34 (bullish bias)
        # 3. Volatility filter
        if (close[i] > dh_aligned[i] and
            close[i] > ema_34_12h_aligned[i] and
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below 12h Donchian low (breakdown)
        # 2. Price below 12h EMA34 (bearish bias)
        # 3. Volatility filter
        elif (close[i] < dl_aligned[i] and
              close[i] < ema_34_12h_aligned[i] and
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_EMA34_VolFilter_v1"
timeframe = "6h"
leverage = 1.0