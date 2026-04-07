#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation
# Uses Donchian channel breakout for entry, 1d EMA(50) for trend filter, and volume spike for confirmation
# Designed for moderate trade frequency (target: 20-40 trades/year) to balance signal quality and fee drag
# Works in bull markets via breakouts and in bear markets via mean reversion at channel extremes

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
    
    # Daily data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA(50) on daily timeframe
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate volume SMA(20) for volume spike detection
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: volume > 1.5x average
        volume_spike = volume[i] > 1.5 * vol_sma[i]
        
        # Long conditions: price breaks above upper Donchian + volume spike + above daily EMA
        if (high[i] > highest_high[i] and volume_spike and 
            close[i] > ema_1d_aligned[i]):
            signals[i] = 0.25
        # Short conditions: price breaks below lower Donchian + volume spike + below daily EMA
        elif (low[i] < lowest_low[i] and volume_spike and 
              close[i] < ema_1d_aligned[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals