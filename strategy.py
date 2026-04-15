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
    
    # Get daily data for 1d HTF context
    daily = get_htf_data(prices, '1d')
    
    # Calculate daily ATR for volatility filter
    tr1 = daily['high'].values[1:] - daily['low'].values[1:]
    tr2 = np.abs(daily['high'].values[1:] - daily['close'].values[:-1])
    tr3 = np.abs(daily['low'].values[1:] - daily['close'].values[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # Volatility filter: ATR > 0.5% of price to avoid low volatility chop
    vol_filter = atr_14d_aligned > (0.005 * close)
    
    # Calculate daily EMA10 for trend direction
    daily_ema_10 = pd.Series(daily['close'].values).ewm(span=10, adjust=False, min_periods=10).mean().values
    daily_ema_10_aligned = align_htf_to_ltf(prices, daily, daily_ema_10)
    
    # Calculate daily EMA30 for additional trend confirmation
    daily_ema_30 = pd.Series(daily['close'].values).ewm(span=30, adjust=False, min_periods=30).mean().values
    daily_ema_30_aligned = align_htf_to_ltf(prices, daily, daily_ema_30)
    
    # Calculate daily volume average for volume spike detection
    vol_ma_4d = pd.Series(daily['volume'].values).rolling(window=4, min_periods=4).mean().values
    vol_ma_4d_aligned = align_htf_to_ltf(prices, daily, vol_ma_4d)
    
    # Volume filter: current volume > 1.3x 4-day average volume
    vol_threshold = 1.3 * vol_ma_4d_aligned
    vol_spike = volume > vol_threshold
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14d_aligned[i]) or np.isnan(daily_ema_10_aligned[i]) or 
            np.isnan(daily_ema_30_aligned[i]) or np.isnan(vol_ma_4d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when volatility is sufficient (avoid chop)
        if not vol_filter[i]:
            signals[i] = 0.0
            continue
        
        # Long: Price above both EMA10 and EMA30 + volume spike
        if (close[i] > daily_ema_10_aligned[i] and 
            close[i] > daily_ema_30_aligned[i] and 
            vol_spike[i]):
            signals[i] = 0.25
        
        # Short: Price below both EMA10 and EMA30 + volume spike
        elif (close[i] < daily_ema_10_aligned[i] and 
              close[i] < daily_ema_30_aligned[i] and 
              vol_spike[i]):
            signals[i] = -0.25
        
        # Exit: reverse signal on opposite direction
        elif (close[i] < daily_ema_10_aligned[i] and signals[i-1] > 0) or \
             (close[i] > daily_ema_10_aligned[i] and signals[i-1] < 0):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_DailyEMA10_30_Volume_Spike_Filter"
timeframe = "12h"
leverage = 1.0