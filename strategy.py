#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout + 1d EMA trend + volume confirmation
# Enter long when: price breaks above Donchian(20) high, EMA50(1d) > EMA200(1d), volume > 1.5x average
# Enter short when: price breaks below Donchian(20) low, EMA50(1d) < EMA200(1d), volume > 1.5x average
# Uses 1d EMA for trend filter to avoid counter-trend trades
# Donchian provides clear breakout levels, volume confirms momentum
# Target: 50-150 total trades over 4 years by requiring multiple confirmations

name = "12h_donchian_1d_ema_vol_v1"
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
    
    # Donchian channels (20-period) on 12h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    
    # EMA on 1d
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR EMA trend flips
            if close[i] <= donchian_low[i] or ema_50_aligned[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR EMA trend flips
            if close[i] >= donchian_high[i] or ema_50_aligned[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + EMA trend + volume
            if volume[i] > volume_threshold[i]:
                # Long: break above Donchian high + uptrend (EMA50 > EMA200)
                if close[i] > donchian_high[i] and ema_50_aligned[i] > ema_200_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: break below Donchian low + downtrend (EMA50 < EMA200)
                elif close[i] < donchian_low[i] and ema_50_aligned[i] < ema_200_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals