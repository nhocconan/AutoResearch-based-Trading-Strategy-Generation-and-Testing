#!/usr/bin/env python3

"""
Hypothesis: 6-hour Donchian(20) breakout with weekly trend filter (1w EMA50) and volume confirmation.
Trade in direction of weekly EMA50 trend when price breaks Donchian(20) channel on 6m with volume spike.
Weekly trend provides directional bias to avoid whipsaws in sideways markets, while Donchian breakout captures
momentum moves. Volume confirmation filters false breakouts. Designed for low trade frequency (12-30/year).
Works in bull markets (follow weekly uptrend longs) and bear markets (follow weekly downtrend shorts).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend direction
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Donchian(20) on 6m: highest high / lowest low of last 20 periods
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: weekly uptrend + price breaks above Donchian high + volume spike
            if ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and close[i] > highest_20[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + price breaks below Donchian low + volume spike
            elif ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and close[i] < lowest_20[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: weekly trend reversal or price returns to opposite Donchian band
            exit_signal = False
            
            if position == 1:
                # Exit long: weekly turns down OR price closes below Donchian low
                if ema50_1w_aligned[i] < ema50_1w_aligned[i-1] or close[i] < lowest_20[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: weekly turns up OR price closes above Donchian high
                if ema50_1w_aligned[i] > ema50_1w_aligned[i-1] or close[i] > highest_20[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_1wEMA50Trend_Volume"
timeframe = "6h"
leverage = 1.0