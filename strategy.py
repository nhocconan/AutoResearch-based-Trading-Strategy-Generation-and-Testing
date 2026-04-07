#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation
# Uses Donchian breakouts for trend entry, 1d EMA50 for trend filter, and 1d volume spike for confirmation.
# Designed for medium trade frequency (target: 20-50 trades/year) to balance signal quality and frequency.
# Works in bull markets via breakout continuation and in bear markets via filtered reversals.

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
    
    # Daily data for trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1d average volume for confirmation
    volume_1d = df_1d['volume'].values
    avg_volume_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average daily volume
        volume_spike = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        # Long: price breaks above Donchian upper band + above 1d EMA50 + volume spike
        if (close[i] > highest_high[i] and 
            close[i] > ema_1d_aligned[i] and 
            volume_spike):
            signals[i] = 0.25
        # Short: price breaks below Donchian lower band + below 1d EMA50 + volume spike
        elif (close[i] < lowest_low[i] and 
              close[i] < ema_1d_aligned[i] and 
              volume_spike):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals