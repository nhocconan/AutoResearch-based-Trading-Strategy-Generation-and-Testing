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
    
    # Get daily data for HTF context (1d)
    daily = get_htf_data(prices, '1d')
    
    # Calculate daily Williams %R (14) for overbought/oversold conditions
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(daily['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(daily['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - daily['close'].values) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    williams_r_aligned = align_htf_to_ltf(prices, daily, williams_r)
    
    # Calculate daily ATR (14) for volatility filter
    tr1 = daily['high'].values[1:] - daily['low'].values[1:]
    tr2 = np.abs(daily['high'].values[1:] - daily['close'].values[:-1])
    tr3 = np.abs(daily['low'].values[1:] - daily['close'].values[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # Volatility filter: ATR > 0.3% of price to avoid low volatility chop
    vol_filter = atr_14d_aligned > (0.003 * close)
    
    # Calculate daily EMA20 for trend direction
    daily_ema_20 = pd.Series(daily['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    daily_ema_20_aligned = align_htf_to_ltf(prices, daily, daily_ema_20)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(atr_14d_aligned[i]) or 
            np.isnan(daily_ema_20_aligned[i])):
            continue
        
        # Only trade when volatility is sufficient (avoid chop)
        if not vol_filter[i]:
            signals[i] = 0.0
            continue
            
        # Long: Williams %R oversold (< -80) and price above EMA20
        if (williams_r_aligned[i] < -80 and 
            close[i] > daily_ema_20_aligned[i]):
            signals[i] = 0.25
        
        # Short: Williams %R overbought (> -20) and price below EMA20
        elif (williams_r_aligned[i] > -20 and 
              close[i] < daily_ema_20_aligned[i]):
            signals[i] = -0.25
        
        # Exit: reverse signal on opposite condition
        elif (williams_r_aligned[i] > -50 and signals[i-1] > 0) or \
             (williams_r_aligned[i] < -50 and signals[i-1] < 0):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_DailyWilliamsR_EMA20_Filter"
timeframe = "6h"
leverage = 1.0