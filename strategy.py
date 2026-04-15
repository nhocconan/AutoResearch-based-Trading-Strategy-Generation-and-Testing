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
    
    # Get weekly data for trend context
    weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA20 for trend direction
    weekly_ema_20 = pd.Series(weekly['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema_20_aligned = align_htf_to_ltf(prices, weekly, weekly_ema_20)
    
    # Calculate weekly EMA50 for trend strength
    weekly_ema_50 = pd.Series(weekly['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema_50_aligned = align_htf_to_ltf(prices, weekly, weekly_ema_50)
    
    # Calculate weekly ATR for volatility filter
    tr1 = weekly['high'].values[1:] - weekly['low'].values[1:]
    tr2 = np.abs(weekly['high'].values[1:] - weekly['close'].values[:-1])
    tr3 = np.abs(weekly['low'].values[1:] - weekly['close'].values[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14w_aligned = align_htf_to_ltf(prices, weekly, atr_14w)
    
    # Volatility filter: ATR > 0.5% of price to avoid low volatility chop
    vol_filter = atr_14w_aligned > (0.005 * close)
    
    # Calculate weekly volume average for volume spike detection
    vol_ma_4w = pd.Series(weekly['volume'].values).rolling(window=4, min_periods=4).mean().values
    vol_ma_4w_aligned = align_htf_to_ltf(prices, weekly, vol_ma_4w)
    
    # Volume filter: current volume > 1.8x 4-week average volume
    vol_threshold = 1.8 * vol_ma_4w_aligned
    vol_spike = volume > vol_threshold
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_ema_20_aligned[i]) or np.isnan(weekly_ema_50_aligned[i]) or 
            np.isnan(atr_14w_aligned[i]) or np.isnan(vol_ma_4w_aligned[i])):
            continue
        
        # Only trade when volatility is sufficient (avoid chop)
        if not vol_filter[i]:
            signals[i] = 0.0
            continue
            
        # Long: Price above both weekly EMA20 and EMA50 + volume spike
        if (close[i] > weekly_ema_20_aligned[i] and 
            close[i] > weekly_ema_50_aligned[i] and 
            vol_spike[i]):
            signals[i] = 0.25
        
        # Short: Price below both weekly EMA20 and EMA50 + volume spike
        elif (close[i] < weekly_ema_20_aligned[i] and 
              close[i] < weekly_ema_50_aligned[i] and 
              vol_spike[i]):
            signals[i] = -0.25
        
        # Exit: reverse signal on opposite direction
        elif (close[i] < weekly_ema_20_aligned[i] and signals[i-1] > 0) or \
             (close[i] > weekly_ema_20_aligned[i] and signals[i-1] < 0):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_WeeklyEMA20_50_Volume_Spike_Filter"
timeframe = "1d"
leverage = 1.0