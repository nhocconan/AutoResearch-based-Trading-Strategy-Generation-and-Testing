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
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Williams %R (14)
    highest_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r_14 = -100 * (highest_high_14 - df_1d['close'].values) / (highest_high_14 - lowest_low_14 + 1e-10)
    williams_r_14_aligned = align_htf_to_ltf(prices, df_1d, williams_r_14)
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_14_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when daily ATR is elevated (> 0.3% of price)
        vol_filter = atr_14_1d_aligned[i] > 0.003 * close[i]
        
        # Long conditions:
        # 1. Price above daily EMA50 (bullish bias)
        # 2. Daily Williams %R oversold (< -80) - mean reversion opportunity
        # 3. Volatility filter
        if (close[i] > ema_50_1d_aligned[i] and
            williams_r_14_aligned[i] < -80 and
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below daily EMA50 (bearish bias)
        # 2. Daily Williams %R overbought (> -20) - mean reversion opportunity
        # 3. Volatility filter
        elif (close[i] < ema_50_1d_aligned[i] and
              williams_r_14_aligned[i] > -20 and
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_DailyWilliamsR_EMA50_VolFilter_v1"
timeframe = "6h"
leverage = 1.0