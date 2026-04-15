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
    
    # Calculate daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Williams %R(14) for momentum/overbought-oversold
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - df_1d['close']) / (highest_high - lowest_low + 1e-10)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r.values)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price above daily EMA34 (bullish bias)
        # 2. Daily Williams %R below -80 (oversold)
        if (close[i] > ema_34_1d_aligned[i] and
            williams_r_aligned[i] < -80):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below daily EMA34 (bearish bias)
        # 2. Daily Williams %R above -20 (overbought)
        elif (close[i] < ema_34_1d_aligned[i] and
              williams_r_aligned[i] > -20):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_EMA34_WilliamsR_v1"
timeframe = "6h"
leverage = 1.0