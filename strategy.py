#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation
# Uses Donchian channel breakouts for entry, 1d EMA for trend filter, and volume spike for confirmation.
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.
# Works in bull markets via trend-following breakouts and in bear markets via mean-reversion at channel extremes.

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
    
    # Daily EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter from daily EMA
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = high[i] > high_max[i-1]  # break above previous high
        short_breakout = low[i] < low_min[i-1]   # break below previous low
        
        # Long: bullish trend + volume spike + long breakout
        if bullish_trend and volume_spike and long_breakout:
            signals[i] = 0.25
        # Short: bearish trend + volume spike + short breakout
        elif bearish_trend and volume_spike and short_breakout:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals