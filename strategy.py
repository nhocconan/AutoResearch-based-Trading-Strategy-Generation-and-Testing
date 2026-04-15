#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d close for calculations
    daily = get_htf_data(prices, '1d')
    close_d = daily['close'].values
    
    # 1d EMA(21) for trend filter
    ema_21d = pd.Series(close_d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21d_aligned = align_htf_to_ltf(prices, daily, ema_21d)
    
    # 1d ATR(14) for volatility filter
    high_d = daily['high'].values
    low_d = daily['low'].values
    tr1 = np.maximum(high_d[1:] - low_d[1:], np.abs(high_d[1:] - close_d[:-1]))
    tr2 = np.maximum(np.abs(low_d[1:] - close_d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # 4h Donchian channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume threshold: 2x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 2.0 * vol_median
    
    # ATR median for volatility regime filter (50-period)
    atr_median = pd.Series(atr_14d_aligned).rolling(window=50, min_periods=50).median()
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21d_aligned[i]) or np.isnan(atr_14d_aligned[i]) or
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(vol_threshold[i]) or np.isnan(atr_median[i])):
            continue
        
        # Volatility filter: avoid extremes (0.5x to 3.0x of median ATR)
        vol_filter = (atr_14d_aligned[i] > 0.5 * atr_median[i]) and (atr_14d_aligned[i] < 3.0 * atr_median[i])
        
        # Long: Price above 1d EMA21 + breaks above 4h Donchian high + volume spike + volatility filter
        if (close[i] > ema_21d_aligned[i] and 
            close[i] > donch_high[i] and 
            volume[i] > vol_threshold[i] and 
            vol_filter):
            signals[i] = 0.25
        
        # Short: Price below 1d EMA21 + breaks below 4h Donchian low + volume spike + volatility filter
        elif (close[i] < ema_21d_aligned[i] and 
              close[i] < donch_low[i] and 
              volume[i] > vol_threshold[i] and 
              vol_filter):
            signals[i] = -0.25
        
        # Exit: price crosses back below/above 1d EMA21
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < ema_21d_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > ema_21d_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_1dEMA21_Donchian20_Vol2x_ATR14dFilter"
timeframe = "4h"
leverage = 1.0