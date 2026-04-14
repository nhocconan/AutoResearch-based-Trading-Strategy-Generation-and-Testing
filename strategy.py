#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with volume confirmation and 12h trend filter
# Donchian channels (20-period high/low) capture breakouts of volatility expansion
# Volume > 1.8x average confirms breakout strength (prevents fakeouts)
# 12h EMA(20) filter ensures we trade in direction of higher timeframe trend
# Exit when price crosses opposite Donchian band (trailing stop within trend)
# Target: 20-40 trades/year per symbol to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA(20)
    ema_len = 20
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=ema_len, adjust=False, min_periods=ema_len).values
    
    # Align EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Donchian Channels (20-period)
    dc_len = 20
    upper_dc = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().values
    lower_dc = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().values
    
    # Volume average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, dc_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_12h_aligned[i]) or 
            np.isnan(upper_dc[i]) or 
            np.isnan(lower_dc[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA
        price_above_ema = close[i] > ema_12h_aligned[i]
        price_below_ema = close[i] < ema_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.8x average
        volume_confirmed = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Enter long: price breaks above upper DC + volume + above 12h EMA
            if (close[i] > upper_dc[i-1] and 
                volume_confirmed and 
                price_above_ema):
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below lower DC + volume + below 12h EMA
            elif (close[i] < lower_dc[i-1] and 
                  volume_confirmed and 
                  price_below_ema):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below lower DC (trailing stop)
            if close[i] < lower_dc[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above upper DC (trailing stop)
            if close[i] > upper_dc[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_Volume_12hEMA_v1"
timeframe = "4h"
leverage = 1.0