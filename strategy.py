#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# Uses Donchian breakout on 1d timeframe for breakout signals:
# - Buy when price breaks above 20-day high in 1w uptrend
# - Sell when price breaks below 20-day low in 1w downtrend
# - 1w EMA100 filter ensures trades align with higher timeframe trend
# - Volume confirmation (current volume > 20-period average) avoids false signals
# Designed for low frequency (target: 8-20 trades/year) to minimize fee drag
# Donchian breakouts are effective in trending markets and capture strong moves

name = "1d_donchian20_1w_ema_volume_v1"
timeframe = "1d"
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
    
    # 1w EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_100_1w = pd.Series(close_1w).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_100_1w)
    
    # Donchian channels (20-period) on 1d data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_100_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter from 1w EMA
        uptrend = close[i] > ema_100_1w_aligned[i]
        downtrend = close[i] < ema_100_1w_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # Break above previous period high
        breakout_down = close[i] < lowest_low[i-1]   # Break below previous period low
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit when price breaks below Donchian low or trend changes
            if close[i] < lowest_low[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high or trend changes
            if close[i] > highest_high[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Entry conditions with trend and volume confirmation
            # Buy when breaking out upward in uptrend
            if breakout_up and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Sell when breaking out downward in downtrend
            elif breakout_down and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals