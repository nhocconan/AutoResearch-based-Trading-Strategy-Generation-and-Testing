#!/usr/bin/env python3

"""
Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and volume confirmation.
Long when price breaks above Donchian(20) high and 12h EMA50 is rising; short when price breaks below Donchian(20) low and 12h EMA50 is falling.
Volume must be above 1.5x 20-period average to confirm breakout strength.
Exit when price returns to Donchian middle or opposite breakout occurs.
Designed for low trade frequency (20-50/year) with strong trend following in both bull and bear markets.
"""

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
    
    # Donchian Channel (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # 12h EMA50 for trend direction
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + 12h EMA50 rising + volume spike
            if close[i] > donchian_high[i] and ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + 12h EMA50 falling + volume spike
            elif close[i] < donchian_low[i] and ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to Donchian middle or opposite breakout
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to middle or breaks below low
                if close[i] <= donchian_mid[i] or close[i] < donchian_low[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to middle or breaks above high
                if close[i] >= donchian_mid[i] or close[i] > donchian_high[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0