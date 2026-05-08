#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with 12h Trend and Volume Spike
# - Donchian(20) breakout above/below 20-period high/low with 12h EMA50 trend filter
# - Volume spike (1.5x 20-period average) confirms breakout strength
# - Works in bull/bear by using 12h EMA50 trend to avoid counter-trend trades
# - Exit on opposite Donchian break (20-period low for longs, high for shorts)
# - Target: 25-40 trades/year to minimize fee drag on 4h timeframe

name = "4h_DonchianBreakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian channels (20-period)
    high_max20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_max20[i]) or np.isnan(low_min20[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + 12h uptrend + volume spike
            long_cond = (close[i] > high_max20[i] and 
                        ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below Donchian low + 12h downtrend + volume spike
            short_cond = (close[i] < low_min20[i] and 
                         ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low (20-period low)
            if close[i] < low_min20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high (20-period high)
            if close[i] > high_max20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals