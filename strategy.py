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
    
    # Get daily data for HTF context
    daily = get_htf_data(prices, '1d')
    
    # Calculate ATR on daily for volatility filter
    tr1 = daily['high'].values[1:] - daily['low'].values[1:]
    tr2 = np.abs(daily['high'].values[1:] - daily['close'].values[:-1])
    tr3 = np.abs(daily['low'].values[1:] - daily['close'].values[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # Volatility filter: ATR > 0.2% of price to avoid low volatility chop
    vol_filter = atr_14d_aligned > (0.002 * close)
    
    # Calculate daily EMA20 for trend direction
    daily_ema_20 = pd.Series(daily['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    daily_ema_20_aligned = align_htf_to_ltf(prices, daily, daily_ema_20)
    
    # Calculate daily EMA50 for additional trend confirmation
    daily_ema_50 = pd.Series(daily['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema_50_aligned = align_htf_to_ltf(prices, daily, daily_ema_50)
    
    # Calculate 4-period EMA of daily volume for volume spike detection
    vol_ema_4d = pd.Series(daily['volume'].values).ewm(span=4, adjust=False, min_periods=4).mean().values
    vol_ema_4d_aligned = align_htf_to_ltf(prices, daily, vol_ema_4d)
    
    # Volume filter: current volume > 1.8x 4-day average volume (more selective)
    vol_threshold = 1.8 * vol_ema_4d_aligned
    vol_spike = volume > vol_threshold
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14d_aligned[i]) or np.isnan(vol_ema_4d_aligned[i]) or 
            np.isnan(daily_ema_20_aligned[i]) or np.isnan(daily_ema_50_aligned[i])):
            continue
        
        # Only trade when volatility is sufficient (avoid chop)
        if not vol_filter[i]:
            signals[i] = 0.0
            continue
            
        # Long: Price above both EMA20 and EMA50 + volume spike
        if (close[i] > daily_ema_20_aligned[i] and 
            close[i] > daily_ema_50_aligned[i] and 
            vol_spike[i]):
            signals[i] = 0.25
        
        # Short: Price below both EMA20 and EMA50 + volume spike
        elif (close[i] < daily_ema_20_aligned[i] and 
              close[i] < daily_ema_50_aligned[i] and 
              vol_spike[i]):
            signals[i] = -0.25
        
        # Exit: reverse signal on opposite direction
        elif (close[i] < daily_ema_20_aligned[i] and signals[i-1] > 0) or \
             (close[i] > daily_ema_20_aligned[i] and signals[i-1] < 0):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_EMA20_50_Volume_Spike_Filter_v2"
timeframe = "4h"
leverage = 1.0