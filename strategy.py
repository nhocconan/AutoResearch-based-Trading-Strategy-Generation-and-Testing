#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high with 12h EMA50 uptrend and volume > 1.5x average
# Short when price breaks below Donchian(20) low with 12h EMA50 downtrend and volume > 1.5x average
# Exit when price crosses the 12h EMA50 in the opposite direction
# Combines price channel breakout with trend and volume filters to reduce false signals
# Designed for 4h timeframe with 12h trend filter - targeting 20-40 trades/year per symbol

name = "4h_Donchian_Breakout_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, EMA50 uptrend, volume spike
            if (close[i] > donch_high[i] and
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, EMA50 downtrend, volume spike
            elif (close[i] < donch_low[i] and
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 12h EMA50
            if close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 12h EMA50
            if close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals