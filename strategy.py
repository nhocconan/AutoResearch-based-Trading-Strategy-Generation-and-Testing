#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (>1.5x 20-bar avg).
# Designed for BTC/ETH robustness: Donchian captures structural breaks, EMA50 ensures trend alignment,
# volume confirms institutional participation. Uses discrete position sizing (0.25) to minimize fee drag.
# Targets 12-37 trades/year on 12h timeframe.

name = "12h_Donchian20_Breakout_1dEMA50_VolumeConfirm_v2"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-bar average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Calculate Donchian channels for breakout (20-period, exclude current bar)
            lookback_high = np.max(high[i-20:i]) if i >= 20 else np.nan
            lookback_low = np.min(low[i-20:i]) if i >= 20 else np.nan
            
            if np.isnan(lookback_high) or np.isnan(lookback_low):
                signals[i] = 0.0
                continue
            
            # LONG: Price breaks above 20-bar high, price > 1d EMA50, volume spike (>1.5x avg)
            if (close[i] > lookback_high and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 20-bar low, price < 1d EMA50, volume spike (>1.5x avg)
            elif (close[i] < lookback_low and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price retests 20-bar low OR loses 1d EMA50 trend
            lookback_low = np.min(low[i-20:i]) if i >= 20 else np.nan
            if (not np.isnan(lookback_low) and 
                (close[i] <= lookback_low or 
                 close[i] < ema_50_1d_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price retests 20-bar high OR gains 1d EMA50 trend
            lookback_high = np.max(high[i-20:i]) if i >= 20 else np.nan
            if (not np.isnan(lookback_high) and 
                (close[i] >= lookback_high or 
                 close[i] > ema_50_1d_aligned[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals