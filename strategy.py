#!/usr/bin/env python3

# Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and volume spike confirmation.
# Works in bull (breakouts with trend) and bear (false breakouts filtered by trend/volume).
# Target: 20-40 trades/year to avoid fee drag.

name = "4h_Donchian20_Breakout_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4-period volume moving average (on 4h data)
    vol_ma_4 = np.full(n, np.nan)
    for i in range(4, n):
        vol_ma_4[i] = np.mean(volume[i-4:i])
    vol_filter = volume > (1.5 * vol_ma_4)  # Volume spike: 1.5x recent average
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    for i in range(20, n):
        highest_20[i] = np.max(high[i-20:i])
        lowest_20[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # Prevent immediate re-entry (8 hours for 4h)
    
    start_idx = max(20, 50, 4)  # Warmup for Donchian, EMA, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: break above Donchian high in 12h uptrend with volume spike
            if (close[i] > highest_20[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: break below Donchian low in 12h downtrend with volume spike
            elif (close[i] < lowest_20[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: price closes below Donchian low OR 12h trend turns down
            if (close[i] < lowest_20[i] or close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above Donchian high OR 12h trend turns up
            if (close[i] > highest_20[i] or close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals