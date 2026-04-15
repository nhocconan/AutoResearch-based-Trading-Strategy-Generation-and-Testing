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
    
    # Get daily data for 1d HTF context
    daily = get_htf_data(prices, '1d')
    
    # Calculate daily ATR for volatility filter
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
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14d_aligned[i]) or np.isnan(vol_ema_4d_aligned[i]) or 
            np.isnan(daily_ema_20_aligned[i]) or np.isnan(daily_ema_50_aligned[i])):
            signals[i] = 0
            continue
        
        # Skip if outside trading session
        if not session_filter[i]:
            signals[i] = 0
            position = 0
            continue
        
        # Only trade when volatility is sufficient (avoid chop)
        if not vol_filter[i]:
            signals[i] = 0
            position = 0
            continue
        
        # Long conditions: Price above both EMA20 and EMA50 + volume spike
        if (close[i] > daily_ema_20_aligned[i] and 
            close[i] > daily_ema_50_aligned[i] and 
            vol_spike[i] and position != 1):
            signals[i] = 0.20
            position = 1
        
        # Short conditions: Price below both EMA20 and EMA50 + volume spike
        elif (close[i] < daily_ema_20_aligned[i] and 
              close[i] < daily_ema_50_aligned[i] and 
              vol_spike[i] and position != -1):
            signals[i] = -0.20
            position = -1
        
        # Exit conditions: reverse signal on opposite direction
        elif (close[i] < daily_ema_20_aligned[i] and position == 1) or \
             (close[i] > daily_ema_20_aligned[i] and position == -1):
            signals[i] = 0.0
            position = 0
        
        # Otherwise, hold current position
        else:
            signals[i] = position * 0.20 if position != 0 else 0.0
    
    return signals

name = "1h_EMA20_50_Volume_Spike_Session_Filter"
timeframe = "1h"
leverage = 1.0