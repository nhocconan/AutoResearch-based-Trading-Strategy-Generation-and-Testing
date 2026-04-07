#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian(20) breakout with 12h trend filter and volume confirmation
# Long when price breaks above 4h Donchian high + 12h EMA > 12h EMA(50) + volume > 1.5x 20-period average
# Short when price breaks below 4h Donchian low + 12h EMA < 12h EMA(50) + volume > 1.5x 20-period average
# Exit when price touches opposite Donchian band
# Uses discrete position sizing (0.25) to minimize churn
# Target: 20-50 trades/year (~80-200 total over 4 years)
name = "4h_donchian20_12h_ema_volume_v1"
timeframe = "4h"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(20) and EMA(50) for trend
    close_12h = df_12h['close'].values
    ema_12h_20 = pd.Series(close_12h).ewm(span=20, adjust=False).mean().values
    ema_12h_50 = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_12h_20_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_20)
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_12h_20_aligned[i]) or np.isnan(ema_12h_50_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price touches opposite band (Donchian low)
            if close[i] <= lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price touches opposite band (Donchian high)
            if close[i] >= highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Enter long: price breaks above Donchian high + 12h EMA20 > EMA50 + volume confirmation
            if (close[i] > highest_high[i] and 
                ema_12h_20_aligned[i] > ema_12h_50_aligned[i] and 
                vol_confirm):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low + 12h EMA20 < EMA50 + volume confirmation
            elif (close[i] < lowest_low[i] and 
                  ema_12h_20_aligned[i] < ema_12h_50_aligned[i] and 
                  vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals