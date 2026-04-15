#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with weekly EMA filter and volume confirmation
# Works in bull markets via breakout continuation, in bear markets via mean reversion off weekly EMA
# Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag
# Primary TF: 1d, HTF: 1w

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
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily HTF data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian to 1d (no shift needed as we're on same TF)
    upper_20_1d = upper_20  # Already aligned to daily timeframe
    lower_20_1d = lower_20  # Already aligned to daily timeframe
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_1d[i]) or np.isnan(lower_20_1d[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Daily price breaks above 20-period Donchian upper
        # 2. Price above weekly EMA50 (bullish bias)
        # 3. Volume confirmation: volume > 1.5x average
        if (close[i] > upper_20_1d[i] and
            close[i] > ema_50_1w_aligned[i] and
            volume_ratio[i] > 1.5):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Daily price breaks below 20-period Donchian lower
        # 2. Price below weekly EMA50 (bearish bias)
        # 3. Volume confirmation: volume > 1.5x average
        elif (close[i] < lower_20_1d[i] and
              close[i] < ema_50_1w_aligned[i] and
              volume_ratio[i] > 1.5):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_1w_Donchian20_1w_EMA50_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0