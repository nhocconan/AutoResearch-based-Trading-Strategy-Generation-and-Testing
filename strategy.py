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
    
    # Weekly SMA(34) for trend filter
    weekly = get_htf_data(prices, '1w')
    close_w = weekly['close'].values
    sma_34w = pd.Series(close_w).rolling(window=34, min_periods=34).mean().values
    sma_34w_aligned = align_htf_to_ltf(prices, weekly, sma_34w)
    
    # Daily ATR(14) for volatility filter and position sizing
    daily = get_htf_data(prices, '1d')
    high_d = daily['high'].values
    low_d = daily['low'].values
    close_d = daily['close'].values
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # Volume threshold: 1.5x median of last 20 days
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(sma_34w_aligned[i]) or np.isnan(atr_14d_aligned[i]) or
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: Close above weekly SMA34 + volume spike
        if close[i] > sma_34w_aligned[i] and volume[i] > vol_threshold[i]:
            signals[i] = 0.25
        # Short: Close below weekly SMA34 + volume spike
        elif close[i] < sma_34w_aligned[i] and volume[i] > vol_threshold[i]:
            signals[i] = -0.25
        # Exit: Price crosses back through weekly SMA34
        elif i > 0 and ((signals[i-1] == 0.25 and close[i] < sma_34w_aligned[i]) or
                        (signals[i-1] == -0.25 and close[i] > sma_34w_aligned[i])):
            signals[i] = 0.0
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_WeeklySMA34_VolumeFilter"
timeframe = "1d"
leverage = 1.0