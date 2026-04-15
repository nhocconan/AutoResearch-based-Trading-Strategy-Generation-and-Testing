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
    
    # 1w EMA(50) for primary trend filter
    weekly = get_htf_data(prices, '1w')
    close_w = weekly['close'].values
    ema_50w = pd.Series(close_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50w_aligned = align_htf_to_ltf(prices, weekly, ema_50w)
    
    # 1d ATR(14) for volatility filter
    daily = get_htf_data(prices, '1d')
    high_d = daily['high'].values
    low_d = daily['low'].values
    close_d = daily['close'].values
    tr1 = np.maximum(high_d[1:] - low_d[1:], np.abs(high_d[1:] - close_d[:-1]))
    tr2 = np.maximum(np.abs(low_d[1:] - close_d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # 12h Donchian(20) for breakout signals
    high_12h = high.copy()
    low_12h = low.copy()
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min()
    
    # Volume threshold: 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50w_aligned[i]) or np.isnan(atr_14d_aligned[i]) or
            np.isnan(donch_high.iloc[i]) or np.isnan(donch_low.iloc[i]) or
            np.isnan(vol_median.iloc[i])):
            continue
        
        # Long: Price breaks above Donchian high + weekly uptrend + volume spike
        if (close[i] > donch_high.iloc[i] and 
            close[i] > ema_50w_aligned[i] and 
            volume[i] > vol_threshold.iloc[i]):
            signals[i] = 0.25
        
        # Short: Price breaks below Donchian low + weekly downtrend + volume spike
        elif (close[i] < donch_low.iloc[i] and 
              close[i] < ema_50w_aligned[i] and 
              volume[i] > vol_threshold.iloc[i]):
            signals[i] = -0.25
        
        # Exit: price crosses back inside Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < donch_high.iloc[i]) or
               (signals[i-1] == -0.25 and close[i] > donch_low.iloc[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_WeeklyEMA50_Donchian20_Vol1.5x"
timeframe = "12h"
leverage = 1.0