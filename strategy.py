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
    
    # Get daily data for context
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    daily_volume = daily['volume'].values
    
    # Calculate daily ATR for volatility
    daily_close_prev = np.concatenate([[daily_close[0]], daily_close[:-1]])
    tr = np.maximum(daily_high - daily_low,
                    np.maximum(np.abs(daily_high - daily_close_prev),
                               np.abs(daily_low - daily_close_prev)))
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily SMA for trend
    sma_50_daily = pd.Series(daily_close).rolling(window=50, min_periods=50).mean().values
    
    # Calculate daily volume SMA for confirmation
    vol_sma_20_daily = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 4h timeframe
    atr_4h = align_htf_to_ltf(prices, daily, atr_daily)
    sma_50_4h = align_htf_to_ltf(prices, daily, sma_50_daily)
    vol_sma_20_4h = align_htf_to_ltf(prices, daily, vol_sma_20_daily)
    
    # Calculate 4h Donchian channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_4h[i]) or np.isnan(sma_50_4h[i]) or 
            np.isnan(vol_sma_20_4h[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above daily average
        volume_confirmed = volume[i] > vol_sma_20_4h[i]
        
        # Long: price breaks above Donchian high in uptrend (price > SMA50) with volume
        if (close[i] > highest_high[i] and 
            close[i] > sma_50_4h[i] and 
            volume_confirmed):
            signals[i] = 0.25
        # Short: price breaks below Donchian low in downtrend (price < SMA50) with volume
        elif (close[i] < lowest_low[i] and 
              close[i] < sma_50_4h[i] and 
              volume_confirmed):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_DonchianBreakout_SMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0