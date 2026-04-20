#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE for regime filter (weekly EMA200)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Load daily data for entry signal (Donchian breakout)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-day Donchian channels on daily data
    highest_20d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian levels to daily timeframe (no alignment needed for daily timeframe)
    highest_20d_aligned = highest_20d  # already daily
    lowest_20d_aligned = lowest_20d    # already daily
    
    # Calculate daily ATR for volatility filter and position sizing
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: daily volume > 20-period average
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(highest_20d_aligned[i]) or np.isnan(lowest_20d_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when price is above weekly EMA200 (bull regime)
        # In bear regime (price < weekly EMA200), we stay flat to avoid losses
        price = close[i]
        bull_regime = price > ema200_1w_aligned[i]
        
        if not bull_regime:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter
        vol_filter = volume[i] > volume_ma_20[i]
        
        # Price levels
        upper_band = highest_20d_aligned[i]
        lower_band = lowest_20d_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above 20-day high with volume
            if price > upper_band and vol_filter:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Long exit: price breaks below 20-day low
            if price < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyEMA200Filter_Volume"
timeframe = "1d"
leverage = 1.0