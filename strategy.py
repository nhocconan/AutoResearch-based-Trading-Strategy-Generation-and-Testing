#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation
# Uses Donchian(20) breakout for entry, 1d EMA(50) for trend filter, and volume spike for confirmation.
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.
# Works in bull markets via trend-following breakouts and in bear via mean-reversion pullbacks.

name = "4h_donchian20_1d_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA(50) on daily data
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_spike = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        # Long: price breaks above upper Donchian band + volume spike + price above daily EMA
        if (high[i] > highest_high[i] and volume_spike and 
            close[i] > ema_1d_aligned[i]):
            signals[i] = 0.25  # 25% position
        
        # Short: price breaks below lower Donchian band + volume spike + price below daily EMA
        elif (low[i] < lowest_low[i] and volume_spike and 
              close[i] < ema_1d_aligned[i]):
            signals[i] = -0.25  # 25% position
        else:
            signals[i] = 0.0
    
    return signals