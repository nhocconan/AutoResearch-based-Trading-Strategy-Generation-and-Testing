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
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(21) for trend filter
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate weekly ATR(14) for volatility regime filter
    tr1_w = df_1w['high'] - df_1w['low']
    tr2_w = np.abs(df_1w['high'] - np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].iloc[:-1]]))
    tr3_w = np.abs(df_1w['low'] - np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].iloc[:-1]]))
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    atr_14_1w = pd.Series(tr_w).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Calculate 6h Donchian(20) channels
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(atr_14_1w_aligned[i]) or 
            np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above/below weekly EMA21
        trend_filter = close[i] > ema_21_1w_aligned[i]
        
        # Weekly volatility regime filter: only trade when weekly ATR is elevated (> 0.8% of price)
        vol_regime = atr_14_1w_aligned[i] > 0.008 * close[i]
        
        # Volume confirmation: volume > 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Long conditions:
        # 1. Price above weekly EMA21 (bullish bias)
        # 2. Price breaks above 6h Donchian(20) high with volume (bullish breakout)
        # 3. Weekly volatility regime filter
        if (trend_filter and
            close[i] > donchian_high_20[i] and
            vol_confirm and
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below weekly EMA21 (bearish bias)
        # 2. Price breaks below 6h Donchian(20) low with volume (bearish breakdown)
        # 3. Weekly volatility regime filter
        elif (not trend_filter and
              close[i] < donchian_low_20[i] and
              vol_confirm and
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyEMA21_ATR_VolRegime_Donchian20_Breakout_v1"
timeframe = "6h"
leverage = 1.0