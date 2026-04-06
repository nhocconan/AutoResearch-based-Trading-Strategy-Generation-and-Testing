#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian(20) breakout with 4h EMA(50) trend filter and volume confirmation
# Enter long when: price breaks above Donchian(20) high, price > 4h EMA(50), volume > 2.0x avg
# Enter short when: price breaks below Donchian(20) low, price < 4h EMA(50), volume > 2.0x avg
# Exit when: price retraces to midpoint of Donchian channel OR opposite breakout occurs
# Uses 4h trend filter to avoid counter-trend trades, targeting 100-200 trades over 4 years
# 1h timeframe balances signal frequency with cost efficiency, using 4h for direction
# Volume confirmation reduces false breakouts, improving win rate

name = "1h_donchian20_4hema50_vol_v1"
timeframe = "1h"
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
    
    # Donchian channel (20-period) on 1h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # 4h EMA(50) for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Volume confirmation: volume > 2.0x 20-period average (strict filter)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price returns to Donchian midpoint OR breaks below lower band
            if close[i] <= donchian_mid[i] or close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price returns to Donchian midpoint OR breaks above upper band
            if close[i] >= donchian_mid[i] or close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: Donchian breakout + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > high_20[i] and close[i] > ema_50_aligned[i]:
                    # Bullish breakout above Donchian high with 4h uptrend
                    signals[i] = 0.20
                    position = 1
                elif close[i] < low_20[i] and close[i] < ema_50_aligned[i]:
                    # Bearish breakdown below Donchian low with 4h downtrend
                    signals[i] = -0.20
                    position = -1
    
    return signals