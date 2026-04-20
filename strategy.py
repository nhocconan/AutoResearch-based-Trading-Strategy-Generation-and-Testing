#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 50-week EMA on weekly data
    ema_50w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50w_aligned = align_htf_to_ltf(prices, df_1w, ema_50w)
    
    # Load daily data ONCE for entry signals
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-day Donchian channels on daily data
    highest_20d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian levels to 1d timeframe (already aligned, but for consistency)
    highest_20d_aligned = align_htf_to_ltf(prices, df_1d, highest_20d)
    lowest_20d_aligned = align_htf_to_ltf(prices, df_1d, lowest_20d)
    
    # Calculate daily ATR for volatility filter
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: daily volume > 20-day average
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in any indicator
        if np.isnan(ema_50w_aligned[i]) or np.isnan(highest_20d_aligned[i]) or np.isnan(lowest_20d_aligned[i]) or np.isnan(atr_1d[i]) or np.isnan(volume_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: only long when price > 50-week EMA, only short when price < 50-week EMA
        price = close[i]
        trend_up = price > ema_50w_aligned[i]
        trend_down = price < ema_50w_aligned[i]
        
        # Volume filter
        vol_filter = volume[i] > volume_ma_20[i]
        
        # Price levels
        upper_band = highest_20d_aligned[i]
        lower_band = lowest_20d_aligned[i]
        
        if position == 0:
            # Long: price breaks above 20-day high with volume and uptrend
            if price > upper_band and vol_filter and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low with volume and downtrend
            elif price < lower_band and vol_filter and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 20-day low
            if price < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 20-day high
            if price > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_50wEMA_Trend_Donchian20_Breakout_VolumeFilter"
timeframe = "1d"
leverage = 1.0