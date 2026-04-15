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
    
    # 12h ATR(14) for volatility filter and stop
    daily = get_htf_data(prices, '12h')
    high_d = daily['high'].values
    low_d = daily['low'].values
    close_d = daily['close'].values
    
    # Calculate True Range
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR calculation with proper min_periods
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, daily, atr_14)
    
    # 12h ATR(50) for long-term volatility regime (avoid low volatility)
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_50_aligned = align_htf_to_ltf(prices, daily, atr_50)
    
    # Volatility filter: ATR(14) > 0.5 * ATR(50) to avoid low vol chop
    vol_regime = atr_14_aligned > (0.5 * atr_50_aligned)
    
    # Volume spike: 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or 
            np.isnan(vol_threshold[i]) or np.isnan(vol_regime[i])):
            continue
        
        # Only trade when volatility regime is favorable (avoid low vol chop)
        if not vol_regime[i]:
            signals[i] = 0.0
            continue
            
        # Long: Close above prior close + volume spike
        if (close[i] > close[i-1] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Close below prior close + volume spike
        elif (close[i] < close[i-1] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: reverse signal on opposite direction
        elif (close[i] < close[i-1] and signals[i-1] > 0) or \
             (close[i] > close[i-1] and signals[i-1] < 0):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_VolRegime_Volume_Momentum"
timeframe = "4h"
leverage = 1.0