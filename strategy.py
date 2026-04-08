#!/usr/bin/env python3
# 12h_donchian_breakout_1d_trend
# Hypothesis: Trade 12h Donchian(20) breakouts in the direction of 1d trend (EMA50).
# Uses 12h price channel breakouts for entry, 1d EMA50 for trend filter, and ATR-based stops.
# Works in bull markets (breakouts with trend) and bear markets (counter-trend bounces at extremes).
# Target: 15-30 trades/year on 12h timeframe with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    # ATR for volatility and stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure EMA50 and ATR are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Calculate 12h Donchian channels (20-period lookback)
        if i >= 20:
            highest_high = np.max(high[i-20:i])
            lowest_low = np.min(low[i-20:i])
        else:
            # Not enough data for Donchian calculation
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian band OR stoploss hit
            if close[i] < lowest_low or close[i] < ema50_aligned[i] - 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian band OR stoploss hit
            if close[i] > highest_high or close[i] > ema50_aligned[i] + 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian band with uptrend (price > EMA50)
            if close[i] > highest_high and close[i] > ema50_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Donchian band with downtrend (price < EMA50)
            elif close[i] < lowest_low and close[i] < ema50_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals