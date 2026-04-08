#!/usr/bin/env python3
# 12h_donchian_breakout_1d_trend_volume_v1
# Hypothesis: On 12h timeframe, Donchian(20) breakout with volume expansion and 1d EMA100 trend alignment captures momentum moves in both bull and bear markets. Trend filter avoids counter-trend entries. Volume confirmation filters false breakouts. Designed for low-frequency, high-conviction trades to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v1"
timeframe = "12h"
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
    
    # Get 1d data for EMA100 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align 1d EMA100 to 12h
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Donchian(20) calculation
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 12h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_100_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price below Donchian low OR price below 1d EMA100
            if (close[i] < donchian_low[i]) or (close[i] < ema_100_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: price above Donchian high OR price above 1d EMA100
            if (close[i] > donchian_high[i]) or (close[i] > ema_100_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: price above Donchian high + volume + price > 1d EMA100
            if (close[i] > donchian_high[i]) and volume_filter[i] and (close[i] > ema_100_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price below Donchian low + volume + price < 1d EMA100
            elif (close[i] < donchian_low[i]) and volume_filter[i] and (close[i] < ema_100_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals