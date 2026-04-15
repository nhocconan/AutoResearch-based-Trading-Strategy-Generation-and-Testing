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
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly ATR(10) for volatility regime
    tr1w = df_1w['high'] - df_1w['low']
    tr2w = np.abs(df_1w['high'] - np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].iloc[:-1]]))
    tr3w = np.abs(df_1w['low'] - np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].iloc[:-1]]))
    tr_1w = np.maximum(tr1w, np.maximum(tr2w, tr3w))
    atr_10_1w = pd.Series(tr_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_10_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_10_1w)
    
    # Calculate weekly EMA(20) for trend direction
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily Donchian(20) channels
    highest_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Calculate daily volume SMA(20) for volume filter
    volume_sma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_10_1w_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or
            np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when weekly ATR is elevated
        vol_regime = atr_10_1w_aligned[i] > 0.008 * close[i]
        
        # Volume confirmation: current 6h volume > 1.5x daily average volume per 6h bar
        # Approximate: daily volume / 4 (since 4x 6h bars per day) * 1.5
        vol_threshold = volume_sma_20_aligned[i] / 4.0 * 1.5
        volume_confirm = volume[i] > vol_threshold
        
        # Long conditions:
        # 1. Price above weekly EMA20 (bullish bias)
        # 2. Price breaks above daily Donchian(20) high (breakout)
        # 3. Volatility regime filter
        # 4. Volume confirmation
        if (close[i] > ema_20_1w_aligned[i] and
            close[i] > highest_20_aligned[i] and
            vol_regime and
            volume_confirm):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below weekly EMA20 (bearish bias)
        # 2. Price breaks below daily Donchian(20) low (breakdown)
        # 3. Volatility regime filter
        # 4. Volume confirmation
        elif (close[i] < ema_20_1w_aligned[i] and
              close[i] < lowest_20_aligned[i] and
              vol_regime and
              volume_confirm):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyEMA20_DailyDonchian20_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0