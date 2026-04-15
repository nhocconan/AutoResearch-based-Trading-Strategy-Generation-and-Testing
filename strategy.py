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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, tr2)])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate daily ATR-based Donchian channel (20 periods)
    atr_scaled_high = high_1d + 0.5 * atr_14
    atr_scaled_low = low_1d - 0.5 * atr_14
    donch_high = pd.Series(atr_scaled_high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(atr_scaled_low).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Volume filter: current volume > 1.5x median of last 20 periods
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
            
        # Long conditions:
        # 1. Price breaks above ATR-adjusted Donchian high
        # 2. Price above daily EMA50 (uptrend filter)
        # 3. Volume confirmation
        if (close[i] > donch_high_aligned[i] and 
            close[i] > ema_50_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below ATR-adjusted Donchian low
        # 2. Price below daily EMA50 (downtrend filter)
        # 3. Volume confirmation
        elif (close[i] < donch_low_aligned[i] and 
              close[i] < ema_50_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
            
        # Exit conditions: reverse signal or volatility drops
        elif (i > 0 and signals[i-1] != 0 and 
              ((signals[i-1] == 0.25 and close[i] < ema_50_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > ema_50_aligned[i]))):
            signals[i] = 0.0
            
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_ATR_Donchian_EMA50_Volume"
timeframe = "12h"
leverage = 1.0