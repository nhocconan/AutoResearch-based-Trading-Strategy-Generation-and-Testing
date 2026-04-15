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
    
    # 1d EMA(50) for trend filter
    daily = get_htf_data(prices, '1d')
    close_d = daily['close'].values
    ema_50d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50d_aligned = align_htf_to_ltf(prices, daily, ema_50d)
    
    # 1d ATR(14) for volatility filter
    high_d = daily['high'].values
    low_d = daily['low'].values
    tr1 = np.maximum(high_d[1:] - low_d[1:], np.abs(high_d[1:] - close_d[:-1]))
    tr2 = np.maximum(np.abs(low_d[1:] - close_d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # Volume threshold: 1.8x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 1.8 * vol_median
    
    # ATR regime filter: avoid low volatility (chop) and extreme volatility
    atr_ma = pd.Series(atr_14d_aligned).rolling(window=30, min_periods=30).mean()
    atr_std = pd.Series(atr_14d_aligned).rolling(window=30, min_periods=30).std()
    vol_regime = (atr_14d_aligned > atr_ma - 0.5 * atr_std) & (atr_14d_aligned < atr_ma + 2.0 * atr_std)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50d_aligned[i]) or np.isnan(atr_14d_aligned[i]) or
            np.isnan(vol_threshold[i]) or np.isnan(vol_regime[i])):
            continue
        
        # Volatility regime filter: avoid low volatility chop and extreme volatility
        if not vol_regime[i]:
            signals[i] = 0.0
            continue
        
        # Long: Price above daily EMA50 + volume spike + volatility regime
        if (close[i] > ema_50d_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.30
        
        # Short: Price below daily EMA50 + volume spike + volatility regime
        elif (close[i] < ema_50d_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.30
        
        # Exit: price crosses back below/above daily EMA50
        elif (i > 0 and 
              ((signals[i-1] == 0.30 and close[i] < ema_50d_aligned[i]) or
               (signals[i-1] == -0.30 and close[i] > ema_50d_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_DailyEMA50_Vol1.8x_VolRegime_v1"
timeframe = "4h"
leverage = 1.0