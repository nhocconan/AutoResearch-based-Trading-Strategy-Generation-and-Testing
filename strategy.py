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
    
    # Weekly EMA(34) for trend filter
    weekly = get_htf_data(prices, '1w')
    close_w = weekly['close'].values
    ema_34w = pd.Series(close_w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34w_aligned = align_htf_to_ltf(prices, weekly, ema_34w)
    
    # Daily Donchian(20) channel
    daily = get_htf_data(prices, '1d')
    high_d = daily['high'].values
    low_d = daily['low'].values
    # Upper band: highest high of last 20 days
    upper_donchian = pd.Series(high_d).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 days
    lower_donchian = pd.Series(low_d).rolling(window=20, min_periods=20).min().values
    upper_donchian_aligned = align_htf_to_ltf(prices, daily, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, daily, lower_donchian)
    
    # Daily ATR(14) for volatility filter and position sizing
    high_d = daily['high'].values
    low_d = daily['low'].values
    close_d = daily['close'].values
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # Volume filter: 1.5x average volume of last 20 days
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_threshold = 1.5 * vol_ma20
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34w_aligned[i]) or np.isnan(upper_donchian_aligned[i]) or
            np.isnan(lower_donchian_aligned[i]) or np.isnan(atr_14d_aligned[i]) or
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: Price breaks above weekly EMA34 AND upper Donchian + volume spike
        if (close[i] > ema_34w_aligned[i] and 
            close[i] > upper_donchian_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Price breaks below weekly EMA34 AND lower Donchian + volume spike
        elif (close[i] < ema_34w_aligned[i] and 
              close[i] < lower_donchian_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price crosses back below/above weekly EMA34
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < ema_34w_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > ema_34w_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_WeeklyEMA34_Donchian20_Vol1.5x"
timeframe = "1d"
leverage = 1.0