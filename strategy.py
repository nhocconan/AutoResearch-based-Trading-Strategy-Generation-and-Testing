#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w trend filter and volume confirmation
# In bull/bear markets: breakout above/below 20-period Donchian channel with volume spike
# Uses 1w EMA(50) for trend filter to avoid counter-trend trades
# Discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Works in trending markets: breakout captures momentum, volume filter avoids false breakouts

name = "12h_1w_donchian_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50) for trend filter
    close_1w_s = pd.Series(close_1w)
    ema_50_1w = close_1w_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h average volume (20-period) for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * avg_volume
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of 1w EMA(50)
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit long if price falls below midpoint of Donchian channel or trend reverses
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < midpoint or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit short if price rises above midpoint of Donchian channel or trend reverses
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > midpoint or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long on breakout above Donchian high with volume confirmation and uptrend
            if close[i] > highest_high[i] and volume_confirmed[i] and uptrend:
                position = 1
                signals[i] = 0.25
            # Enter short on breakout below Donchian low with volume confirmation and downtrend
            elif close[i] < lowest_low[i] and volume_confirmed[i] and downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals