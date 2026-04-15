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
    
    # Load 1d data ONCE before loop
    daily = get_htf_data(prices, '1d')
    high_d = daily['high'].values
    low_d = daily['low'].values
    close_d = daily['close'].values
    volume_d = daily['volume'].values
    
    # 14-day ATR for volatility filter
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # 20-period EMA of daily close for trend filter
    ema_20d = pd.Series(close_d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20d_aligned = align_htf_to_ltf(prices, daily, ema_20d)
    
    # 10-period EMA of daily volume for volume context
    vol_ema_10d = pd.Series(volume_d).ewm(span=10, adjust=False, min_periods=10).mean().values
    vol_ema_10d_aligned = align_htf_to_ltf(prices, daily, vol_ema_10d)
    
    # Volume spike: current volume > 1.5x 10-day EMA of daily volume
    vol_spike = volume > (1.5 * vol_ema_10d_aligned)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14d_aligned[i]) or np.isnan(ema_20d_aligned[i]) or 
            np.isnan(vol_ema_10d_aligned[i])):
            continue
        
        # Volatility filter: ATR > 0.3% of price (avoid low volatility chop)
        if atr_14d_aligned[i] <= (0.003 * close[i]):
            signals[i] = 0.0
            continue
        
        # Long: Price above 20-day EMA + volume spike
        if (close[i] > ema_20d_aligned[i] and vol_spike[i]):
            signals[i] = 0.25
        
        # Short: Price below 20-day EMA + volume spike
        elif (close[i] < ema_20d_aligned[i] and vol_spike[i]):
            signals[i] = -0.25
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_EMA20_Volume_Spike_Filter"
timeframe = "12h"
leverage = 1.0