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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter (weekly bias proxy)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    df_1d_tr = np.maximum(
        df_1d['high'].values - df_1d['low'].values,
        np.maximum(
            np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]])),
            np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))
        )
    )
    atr_14_1d = pd.Series(df_1d_tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 4h Donchian(20) channels for breakout signals
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Daily trend filter: price above 1d EMA50 (bullish daily bias)
        # 2. Price breaks above 4h Donchian(20) high with volume confirmation
        # 3. Volume > 1.3x average
        # 4. Adequate volatility: ATR > 0.25% of price
        if (close[i] > ema_50_1d_aligned[i] and
            close[i] > donchian_high[i] and
            volume_ratio[i] > 1.3 and
            atr_14_1d_aligned[i] > 0.0025 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Daily trend filter: price below 1d EMA50 (bearish daily bias)
        # 2. Price breaks below 4h Donchian(20) low with volume confirmation
        # 3. Volume > 1.3x average
        # 4. Adequate volatility: ATR > 0.25% of price
        elif (close[i] < ema_50_1d_aligned[i] and
              close[i] < donchian_low[i] and
              volume_ratio[i] > 1.3 and
              atr_14_1d_aligned[i] > 0.0025 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_EMA50_Donchian20_Breakout_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0