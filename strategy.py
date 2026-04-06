#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation
# Long: price breaks above Donchian(20) high, price > 1d EMA(50), volume > 2x average
# Short: price breaks below Donchian(20) low, price < 1d EMA(50), volume > 2x average
# Exit: price returns to Donchian(20) midpoint or opposite breakout occurs
# Uses daily trend to filter counter-trend trades, targeting 50-150 trades over 4 years

name = "12h_donchian20_1dema_vol_v1"
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
    
    # Donchian(20) on 12h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donch_high = high_roll.values
    donch_low = low_roll.values
    donch_mid = (donch_high + donch_low) / 2
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price returns to midpoint OR breaks below lower band
            if close[i] <= donch_mid[i] or close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to midpoint OR breaks above upper band
            if close[i] >= donch_mid[i] or close[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout + trend filter + volume
            if close[i] > donch_high[i] and close[i] > ema_50_aligned[i] and volume[i] > volume_threshold[i]:
                # Bullish breakout above Donchian high with daily uptrend
                signals[i] = 0.25
                position = 1
            elif close[i] < donch_low[i] and close[i] < ema_50_aligned[i] and volume[i] > volume_threshold[i]:
                # Bearish breakdown below Donchian low with daily downtrend
                signals[i] = -0.25
                position = -1
    
    return signals