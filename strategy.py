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
    
    # Calculate daily Williams %R(14) for momentum extreme
    highest_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r_14_1d = -100 * (highest_high_14 - df_1d['close'].values) / (highest_high_14 - lowest_low_14 + 1e-10)
    williams_r_14_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_14_1d)
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h ATR(14) for volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.concatenate([[close[0]], close[:-1]])), np.abs(low - np.concatenate([[close[0]], close[:-1]]))))
    atr_14_6h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_14_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_14_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when 6h ATR is elevated (> 0.8% of price)
        vol_filter = atr_14_6h[i] > 0.008 * close[i]
        
        # Long conditions:
        # 1. Price above daily EMA50 (bullish bias)
        # 2. Daily Williams %R oversold (< -80) - extreme pessimism
        # 3. Volatility filter
        if (close[i] > ema_50_1d_aligned[i] and
            williams_r_14_1d_aligned[i] < -80 and
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below daily EMA50 (bearish bias)
        # 2. Daily Williams %R overbought (> -20) - extreme optimism
        # 3. Volatility filter
        elif (close[i] < ema_50_1d_aligned[i] and
              williams_r_14_1d_aligned[i] > -20 and
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_EMA50_VolFilter_v1"
timeframe = "6h"
leverage = 1.0