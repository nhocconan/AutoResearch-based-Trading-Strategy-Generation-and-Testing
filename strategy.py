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
    
    # Weekly Donchian channel (55 periods)
    df_1w = get_htf_data(prices, '1w')
    donch_high = pd.Series(df_1w['high']).rolling(window=55, min_periods=55).max().values
    donch_low = pd.Series(df_1w['low']).rolling(window=55, min_periods=55).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # Daily ATR for volatility filter (14 periods)
    df_1d = get_htf_data(prices, '1d')
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Weekly volume average for confirmation
    vol_ma = pd.Series(df_1w['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            continue
        
        # Long: break above weekly Donchian high + volume > 1.5x weekly average
        if (close[i] > donch_high_aligned[i] and 
            volume[i] > 1.5 * vol_ma_aligned[i]):
            signals[i] = 0.25
        
        # Short: break below weekly Donchian low + volume > 1.5x weekly average
        elif (close[i] < donch_low_aligned[i] and 
              volume[i] > 1.5 * vol_ma_aligned[i]):
            signals[i] = -0.25
        
        # Exit: price crosses back inside the Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < donch_high_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > donch_low_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_WeeklyDonchian_Volume_Breakout"
timeframe = "12h"
leverage = 1.0