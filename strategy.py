#!/usr/bin/env python3
# Hypothesis: 12h Donchian channel breakout with 1d EMA34 trend filter and volume confirmation.
# Uses 1d EMA34 for trend alignment, 12h Donchian(20) levels for breakout entry, and volume spike (>1.5x 20-bar avg) for confirmation.
# Designed for low trade frequency (target 50-150 total over 4 years) to minimize fee drag and improve test generalization.
# Works in both bull and bear markets by following the 1d trend direction and requiring volume confirmation to avoid false breakouts.

name = "12h_Donchian20_Breakout_1dEMA34_VolumeConfirm_v1"
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian levels from prior 12h bar (primary TF)
    lookback = 20
    # Donchian Upper = max(high, lookback), Lower = min(low, lookback)
    # Using prior bar's OHLC to avoid look-ahead
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    donchian_upper = pd.Series(prev_high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_lower = pd.Series(prev_low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate average volume for confirmation (20-period LTF)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper, close > 1d EMA34, volume spike (>1.5x avg)
            if (high[i] > donchian_upper[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower, close < 1d EMA34, volume spike (>1.5x avg)
            elif (low[i] < donchian_lower[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price breaks below Donchian lower or volume drops
            if (low[i] < donchian_lower[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if price breaks above Donchian upper or volume drops
            if (high[i] > donchian_upper[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals