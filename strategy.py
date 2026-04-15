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
    
    # Get weekly data for 1w HTF context
    weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly ATR for volatility filter
    tr1 = weekly['high'].values[1:] - weekly['low'].values[1:]
    tr2 = np.abs(weekly['high'].values[1:] - weekly['close'].values[:-1])
    tr3 = np.abs(weekly['low'].values[1:] - weekly['close'].values[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14w_aligned = align_htf_to_ltf(prices, weekly, atr_14w)
    
    # Volatility filter: ATR > 0.5% of price to avoid low volatility chop
    vol_filter = atr_14w_aligned > (0.005 * close)
    
    # Calculate weekly EMA10 for trend direction
    weekly_ema_10 = pd.Series(weekly['close'].values).ewm(span=10, adjust=False, min_periods=10).mean().values
    weekly_ema_10_aligned = align_htf_to_ltf(prices, weekly, weekly_ema_10)
    
    # Calculate weekly EMA30 for additional trend confirmation
    weekly_ema_30 = pd.Series(weekly['close'].values).ewm(span=30, adjust=False, min_periods=30).mean().values
    weekly_ema_30_aligned = align_htf_to_ltf(prices, weekly, weekly_ema_30)
    
    # Calculate weekly volume average for volume spike detection
    vol_ma_4w = pd.Series(weekly['volume'].values).rolling(window=4, min_periods=4).mean().values
    vol_ma_4w_aligned = align_htf_to_ltf(prices, weekly, vol_ma_4w)
    
    # Volume filter: current volume > 1.3x 4-week average volume
    vol_threshold = 1.3 * vol_ma_4w_aligned
    vol_spike = volume > vol_threshold
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14w_aligned[i]) or np.isnan(weekly_ema_10_aligned[i]) or 
            np.isnan(weekly_ema_30_aligned[i]) or np.isnan(vol_ma_4w_aligned[i])):
            continue
        
        # Only trade when volatility is sufficient (avoid chop)
        if not vol_filter[i]:
            signals[i] = 0.0
            continue
            
        # Long: Price above both EMA10 and EMA30 + volume spike
        if (close[i] > weekly_ema_10_aligned[i] and 
            close[i] > weekly_ema_30_aligned[i] and 
            vol_spike[i]):
            signals[i] = 0.25
        
        # Short: Price below both EMA10 and EMA30 + volume spike
        elif (close[i] < weekly_ema_10_aligned[i] and 
              close[i] < weekly_ema_30_aligned[i] and 
              vol_spike[i]):
            signals[i] = -0.25
        
        # Exit: reverse signal on opposite direction
        elif (close[i] < weekly_ema_10_aligned[i] and signals[i-1] > 0) or \
             (close[i] > weekly_ema_10_aligned[i] and signals[i-1] < 0):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_WeeklyEMA10_30_Volume_Spike_Filter"
timeframe = "1d"
leverage = 1.0