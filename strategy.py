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
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d HTF data for weekly ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d ATR percentile rank (20-period) for regime filter
    atr_percentile = pd.Series(atr_14_1d).rolling(window=20, min_periods=20).rank(pct=True).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Calculate 1h Donchian channels (20-period) for entry timing
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Session filter: 08-20 UTC (avoid Asian session noise)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in low volatility environments (ATR percentile < 0.3)
        # This avoids choppy markets and focuses on clear breakouts
        if atr_percentile_aligned[i] >= 0.3:
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 1h price breaks above 20-period Donchian upper
        # 2. 4h EMA(50) uptrend (price above EMA)
        # 3. Volume confirmation: volume > 1.5x average
        if (close[i] > high_roll[i] and
            close[i] > ema_50_4h_aligned[i] and
            volume_ratio[i] > 1.5):
            signals[i] = 0.20
            
        # Short conditions:
        # 1. 1h price breaks below 20-period Donchian lower
        # 2. 4h EMA(50) downtrend (price below EMA)
        # 3. Volume confirmation: volume > 1.5x average
        elif (close[i] < low_roll[i] and
              close[i] < ema_50_4h_aligned[i] and
              volume_ratio[i] > 1.5):
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4h_EMA50_Donchian20_Volume_ATRRegime_v1"
timeframe = "1h"
leverage = 1.0