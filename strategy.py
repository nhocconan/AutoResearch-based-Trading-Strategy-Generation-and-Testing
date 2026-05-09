#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 12h EMA50 trend filter.
# Uses 4h Donchian breakouts for entry, volume surge for confirmation, and 12h EMA50 to
# align with higher timeframe trend. Designed to capture breakouts in both bull and bear
# markets while avoiding false breakouts in ranging conditions. Target: 20-50 trades/year.
name = "4h_Donchian20_Volume_12hEMA50_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 12h close
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period moving average of volume for volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period high/low)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        ema = ema_50_12h_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        vol_avg = vol_ma[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian channel + volume surge + above 12h EMA50
            if price > upper_channel and vol > 1.5 * vol_avg and price > ema:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian channel + volume surge + below 12h EMA50
            elif price < lower_channel and vol > 1.5 * vol_avg and price < ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower Donchian channel OR below 12h EMA50
            if price < lower_channel or price < ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper Donchian channel OR above 12h EMA50
            if price > upper_channel or price > ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals