#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA(20) trend filter and volume confirmation
# Uses Donchian breakouts for trend following, confirmed by 12h EMA trend and volume above average.
# Designed for moderate trade frequency (target: 20-50 trades/year) with clear trend signals.
# Works in bull markets via breakout longs and in bear markets via breakdown shorts.
# Volume filter reduces false breakouts during low conviction periods.

name = "6h_donchian20_12h_ema_volume_v1"
timeframe = "6h"
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
    
    # 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA(20) for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_avg[i]
        
        # Long conditions: price breaks above Donchian upper band + uptrend + volume
        if (close[i] > highest_high[i] and 
            close[i] > ema_12h_aligned[i] and 
            vol_confirm):
            signals[i] = 0.25
        
        # Short conditions: price breaks below Donchian lower band + downtrend + volume
        elif (close[i] < lowest_low[i] and 
              close[i] < ema_12h_aligned[i] and 
              vol_confirm):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals