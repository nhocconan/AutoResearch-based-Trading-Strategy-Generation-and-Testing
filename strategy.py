#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian breakout with daily EMA trend filter and volume confirmation
# Uses Donchian(20) channels for breakout signals, daily EMA(50) for trend direction,
# and volume > 1.5x 20-period average for confirmation. Designed for low trade frequency
# (target: 15-35 trades/year) to minimize fee drag. Works in bull markets via breakout
# continuation and in bear markets via mean reversion at channel extremes.

name = "12h_donchian20_daily_ema_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Calculate daily EMA(50)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long: price breaks above upper Donchian band + above daily EMA + volume confirmed
        if (close[i] > highest_high[i] and 
            close[i] > ema_1d_aligned[i] and 
            vol_confirmed[i]):
            signals[i] = 0.25
        # Short: price breaks below lower Donchian band + below daily EMA + volume confirmed
        elif (close[i] < lowest_low[i] and 
              close[i] < ema_1d_aligned[i] and 
              vol_confirmed[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals