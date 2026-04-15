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
    
    # Calculate daily ATR for volatility filter
    daily_close_prev = np.concatenate([[daily_close[0]], daily_close[:-1]])
    tr = np.maximum(daily_high - daily_low,
                    np.maximum(np.abs(daily_high - daily_close_prev),
                               np.abs(daily_low - daily_close_prev)))
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily Bollinger Bands
    sma_20 = pd.Series(daily_close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(daily_close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    
    # Calculate 20-day volume average
    vol_ma_20 = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    upper_band_4h = align_htf_to_ltf(prices, daily, upper_band)
    lower_band_4h = align_htf_to_ltf(prices, daily, lower_band)
    vol_ma_4h = align_htf_to_ltf(prices, daily, vol_ma_20)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band_4h[i]) or np.isnan(lower_band_4h[i]) or 
            np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current daily volume > 1.5x 20-day average
        volume_spike = daily_volume[i // 24] > (vol_ma_20[i // 24] * 1.5) if i // 24 < len(daily_volume) else False
        
        # Long when price touches lower BB with volume spike
        if (close[i] <= lower_band_4h[i] and volume_spike):
            signals[i] = 0.25
        # Short when price touches upper BB with volume spike
        elif (close[i] >= upper_band_4h[i] and volume_spike):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_DailyBB_Volume_Spike_MeanReversion"
timeframe = "4h"
leverage = 1.0