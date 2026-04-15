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
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    daily_volume = daily['volume'].values
    
    # Calculate 12h Donchian channels (20-period) for breakout signals
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate daily ATR for volatility filter
    daily_close_prev = np.concatenate([[daily_close[0]], daily_close[:-1]])
    tr = np.maximum(daily_high - daily_low,
                    np.maximum(np.abs(daily_high - daily_close_prev),
                               np.abs(daily_low - daily_close_prev)))
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ratio_daily = atr_daily / daily_close
    
    # Align daily ATR ratio to 12h timeframe
    atr_ratio_12h = align_htf_to_ltf(prices, daily, atr_ratio_daily)
    
    # Calculate daily volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio_daily = daily_volume / (vol_ma_20 + 1e-10)
    
    # Align daily volume ratio to 12h timeframe
    volume_ratio_12h = align_htf_to_ltf(prices, daily, volume_ratio_daily)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_ratio_12h[i]) or np.isnan(volume_ratio_12h[i])):
            signals[i] = 0.0
            continue
        
        # Breakout strategy with volatility and volume filters
        # Long when price breaks above Donchian high in high volatility + high volume
        if (close[i] > highest_high[i] and 
            atr_ratio_12h[i] > 0.015 and  # Volatility filter
            volume_ratio_12h[i] > 1.5):   # Volume confirmation
            signals[i] = 0.25
        # Short when price breaks below Donchian low in high volatility + high volume
        elif (close[i] < lowest_low[i] and 
              atr_ratio_12h[i] > 0.015 and  # Volatility filter
              volume_ratio_12h[i] > 1.5):   # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Breakout_Volume_Volatility_Filter"
timeframe = "12h"
leverage = 1.0