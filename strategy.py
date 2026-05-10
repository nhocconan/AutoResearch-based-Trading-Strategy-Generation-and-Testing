#!/usr/bin/env python3
# 4H_12H_PriceChannel_Breakout_Volume
# Hypothesis: Use 12h Donchian channels (20) for breakouts, confirmed by 12h EMA50 trend and 4h volume spike (>1.5x 20-period average). Enter long when price breaks above upper channel in uptrend, short when breaks below lower channel in downtrend. Exit when price crosses the 12h EMA50. Uses volume to filter false breakouts and EMA for trend direction. Target: 20-40 trades/year per symbol.

name = "4H_12H_PriceChannel_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h Donchian channel (20)
    high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA50 for trend
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 4h volume spike (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Align HTF indicators to 4h
    upper_chan = align_htf_to_ltf(prices, df_12h, high_20)
    lower_chan = align_htf_to_ltf(prices, df_12h, low_20)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_chan[i]) or np.isnan(lower_chan[i]) or
            np.isnan(ema50_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_spike = volume_spike_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: uptrend + price breaks above upper channel + volume spike
            if close[i] > ema50_aligned[i] and close[i] > upper_chan[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: downtrend + price breaks below lower channel + volume spike
            elif close[i] < ema50_aligned[i] and close[i] < lower_chan[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below EMA50
            if close[i] < ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA50
            if close[i] > ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals