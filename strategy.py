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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d volume MA(20) for volume confirmation
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to daily EMA34
        trend_filter = close[i] > ema_34_1d_aligned[i]
        
        # Volume confirmation: volume > 1.8x average
        volume_filter = volume[i] > 1.8 * vol_ma_20_aligned[i]
        
        # Long conditions:
        # 1. Price above daily EMA34 (bullish bias)
        # 2. Volume confirmation
        if trend_filter and volume_filter:
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below daily EMA34 (bearish bias)
        # 2. Volume confirmation
        elif (not trend_filter and volume_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_EMA34_Volume_Confirmation_v1"
timeframe = "1d"
leverage = 1.0