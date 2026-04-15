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
    
    # Get daily data for trend and volatility context
    daily = get_htf_data(prices, '1d')
    
    # Calculate ATR on daily for volatility filter and stop loss
    tr1 = daily['high'].values[1:] - daily['low'].values[1:]
    tr2 = np.abs(daily['high'].values[1:] - daily['close'].values[:-1])
    tr3 = np.abs(daily['low'].values[1:] - daily['close'].values[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # Calculate daily EMA50 for trend direction
    daily_ema_50 = pd.Series(daily['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema_50_aligned = align_htf_to_ltf(prices, daily, daily_ema_50)
    
    # Calculate daily EMA200 for long-term trend filter
    daily_ema_200 = pd.Series(daily['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    daily_ema_200_aligned = align_htf_to_ltf(prices, daily, daily_ema_200)
    
    # Calculate 4-period EMA of daily volume for volume spike detection
    vol_ema_4d = pd.Series(daily['volume'].values).ewm(span=4, adjust=False, min_periods=4).mean().values
    vol_ema_4d_aligned = align_htf_to_ltf(prices, daily, vol_ema_4d)
    
    # Volume threshold: current volume > 2.0x 4-day average volume
    vol_threshold = 2.0 * vol_ema_4d_aligned
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14d_aligned[i]) or np.isnan(daily_ema_50_aligned[i]) or 
            np.isnan(daily_ema_200_aligned[i]) or np.isnan(vol_ema_4d_aligned[i])):
            continue
        
        # Volatility filter: ATR > 0.4% of price to avoid low volatility chop
        if atr_14d_aligned[i] <= (0.004 * close[i]):
            signals[i] = 0.0
            continue
        
        # Long: Price above EMA50 AND EMA200 (bullish trend) + volume spike
        if (close[i] > daily_ema_50_aligned[i] and 
            close[i] > daily_ema_200_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Price below EMA50 AND EMA200 (bearish trend) + volume spike
        elif (close[i] < daily_ema_50_aligned[i] and 
              close[i] < daily_ema_200_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: reverse signal on opposite direction
        elif (close[i] < daily_ema_50_aligned[i] and signals[i-1] > 0) or \
             (close[i] > daily_ema_50_aligned[i] and signals[i-1] < 0):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4d_EMA50_200_Volume_Spike_Filter"
timeframe = "4h"
leverage = 1.0