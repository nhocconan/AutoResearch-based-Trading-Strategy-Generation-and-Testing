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
    
    # Get weekly data for HTF trend
    weekly = get_htf_data(prices, '1w')
    weekly_close = weekly['close'].values
    weekly_high = weekly['high'].values
    weekly_low = weekly['low'].values
    
    # Calculate weekly EMA40 for trend filter
    ema_40 = pd.Series(weekly_close).ewm(span=40, adjust=False, min_periods=40).mean().values
    
    # Align weekly EMA to daily timeframe
    ema_40_aligned = align_htf_to_ltf(prices, weekly, ema_40)
    
    # Calculate weekly ATR for volatility regime
    weekly_close_prev = np.concatenate([[weekly_close[0]], weekly_close[:-1]])
    tr = np.maximum(weekly_high - weekly_low,
                    np.maximum(np.abs(weekly_high - weekly_close_prev),
                               np.abs(weekly_low - weekly_close_prev)))
    atr_weekly = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ratio_weekly = atr_weekly / weekly_close
    
    # Align weekly ATR ratio to daily timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, weekly, atr_ratio_weekly)
    
    # Calculate daily Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_40_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending regimes (low volatility = choppy)
        if atr_ratio_aligned[i] > 0.02:  # High volatility = trending regime
            continue
            
        # Long: price breaks above Donchian high in uptrend (price > weekly EMA40)
        if (close[i] > highest_high[i] and 
            close[i] > ema_40_aligned[i]):
            signals[i] = 0.30
        # Short: price breaks below Donchian low in downtrend (price < weekly EMA40)
        elif (close[i] < lowest_low[i] and 
              close[i] < ema_40_aligned[i]):
            signals[i] = -0.30
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyEMA40_Donchian20_VolatilityFilter"
timeframe = "1d"
leverage = 1.0