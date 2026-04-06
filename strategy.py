#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA filter and volume confirmation
# Enter long when price breaks above Donchian(20) high, EMA50 > EMA200 on daily, volume > 1.5x average
# Enter short when price breaks below Donchian(20) low, EMA50 < EMA200 on daily, volume > 1.5x average
# Exit when price crosses the opposite Donchian band or volume dries up
# Trend following with volatility breakout, designed for both bull and bear markets
# Target: 80-180 trades over 4 years (20-45/year)

name = "4h_donchian_1d_ema_vol_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily EMA filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR EMA50 < EMA200
            if close[i] < low_20[i] or ema_50_aligned[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR EMA50 > EMA200
            if close[i] > high_20[i] or ema_50_aligned[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + EMA alignment + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > high_20[i] and ema_50_aligned[i] > ema_200_aligned[i]:
                    # Bullish breakout with uptrend on daily
                    signals[i] = 0.25
                    position = 1
                elif close[i] < low_20[i] and ema_50_aligned[i] < ema_200_aligned[i]:
                    # Bearish breakout with downtrend on daily
                    signals[i] = -0.25
                    position = -1
    
    return signals