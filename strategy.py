#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 1d/1w HTF context
    daily = get_htf_data(prices, '1d')
    weekly = get_htf_data(prices, '1w')
    
    # Calculate ATR on daily for volatility filter
    tr1 = daily['high'].values[1:] - daily['low'].values[1:]
    tr2 = np.abs(daily['high'].values[1:] - daily['close'].values[:-1])
    tr3 = np.abs(daily['low'].values[1:] - daily['close'].values[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # Volatility filter: ATR > 0.3% of price to avoid low volatility chop
    vol_filter = atr_14d_aligned > (0.003 * close)
    
    # Calculate 4-period EMA of daily volume for volume spike detection
    vol_ema_4d = pd.Series(daily['volume'].values).ewm(span=4, adjust=False, min_periods=4).mean().values
    vol_ema_4d_aligned = align_htf_to_ltf(prices, daily, vol_ema_4d)
    
    # Volume filter: current volume > 1.5x 4-day average volume
    vol_threshold = 1.5 * vol_ema_4d_aligned
    vol_spike = volume > vol_threshold
    
    # Calculate daily EMA20 for trend direction
    daily_ema_20 = pd.Series(daily['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    daily_ema_20_aligned = align_htf_to_ltf(prices, daily, daily_ema_20)
    
    # Calculate daily EMA50 for additional trend confirmation
    daily_ema_50 = pd.Series(daily['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema_50_aligned = align_htf_to_ltf(prices, daily, daily_ema_50)
    
    # Calculate weekly EMA20 for higher timeframe trend
    weekly_ema_20 = pd.Series(weekly['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema_20_aligned = align_htf_to_ltf(prices, weekly, weekly_ema_20)
    
    signals = np.zeros(n)
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14d_aligned[i]) or np.isnan(vol_ema_4d_aligned[i]) or 
            np.isnan(daily_ema_20_aligned[i]) or np.isnan(daily_ema_50_aligned[i]) or
            np.isnan(weekly_ema_20_aligned[i])):
            continue
        
        # Only trade when volatility is sufficient (avoid chop)
        if not vol_filter[i]:
            signals[i] = 0.0
            continue
            
        # Long: Price above both daily EMA20 and EMA50 + above weekly EMA20 + volume spike
        if (close[i] > daily_ema_20_aligned[i] and 
            close[i] > daily_ema_50_aligned[i] and
            close[i] > weekly_ema_20_aligned[i] and
            vol_spike[i]):
            signals[i] = 0.25
        
        # Short: Price below both daily EMA20 and EMA50 + below weekly EMA20 + volume spike
        elif (close[i] < daily_ema_20_aligned[i] and 
              close[i] < daily_ema_50_aligned[i] and
              close[i] < weekly_ema_20_aligned[i] and
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

name = "1d_EMA20_50_WeeklyEMA20_Volume_Spike_Filter"
timeframe = "1d"
leverage = 1.0