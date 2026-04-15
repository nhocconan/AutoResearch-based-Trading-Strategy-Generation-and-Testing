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
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 6h Donchian(20) breakout levels
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when daily ATR is elevated (> 0.6% of price)
        vol_filter = atr_14_1d_aligned[i] > 0.006 * close[i]
        
        # Long conditions:
        # 1. Price above 1d EMA(50) (bullish trend)
        # 2. Price breaks above 6h Donchian(20) high with volume
        # 3. Volume confirmation: volume > 1.5x average
        if (close[i] > ema_50_1d_aligned[i] and
            close[i] > donchian_high_20[i] and
            volume_ratio[i] > 1.5 and
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below 1d EMA(50) (bearish trend)
        # 2. Price breaks below 6h Donchian(20) low with volume
        # 3. Volume confirmation: volume > 1.5x average
        elif (close[i] < ema_50_1d_aligned[i] and
              close[i] < donchian_low_20[i] and
              volume_ratio[i] > 1.5 and
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_EMA50_Donchian20_Volume_Breakout_v1"
timeframe = "6h"
leverage = 1.0