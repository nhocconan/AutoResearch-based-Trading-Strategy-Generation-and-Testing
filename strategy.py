#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(21) trend filter and volume confirmation
# Uses Donchian channel breakout for entry, 12h EMA for trend direction, and volume > 1.5x 20-period average for confirmation.
# Designed for moderate trade frequency (target: 20-40 trades/year) to balance opportunity and cost.
# Works in bull markets via breakout longs and in bear markets via breakdown shorts.

name = "4h_donchian20_12h_ema_volume_v2"
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
    
    # 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 25:
        return np.zeros(n)
    
    # Calculate EMA(21) on 12h close
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Donchian channel (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above Donchian upper band + above 12h EMA + volume confirmation
        if (close[i] > highest_high[i] and 
            close[i] > ema_12h_aligned[i] and 
            vol_confirm[i]):
            signals[i] = 0.25
        # Short conditions: price breaks below Donchian lower band + below 12h EMA + volume confirmation
        elif (close[i] < lowest_low[i] and 
              close[i] < ema_12h_aligned[i] and 
              vol_confirm[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals