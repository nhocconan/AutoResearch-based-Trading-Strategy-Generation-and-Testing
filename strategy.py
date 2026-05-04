#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation
# Donchian breakouts capture momentum in trending markets. EMA50 on 1d ensures we only trade
# in the direction of the higher timeframe trend. Volume spike (>1.5x 20-period EMA) confirms
# participation. Designed for 4h timeframe targeting 20-50 trades per year per symbol.
# Uses discrete position sizing (0.30) to balance return potential and drawdown control.

name = "4h_Donchian20_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Look for Donchian breakouts with trend and volume confirmation
            if close[i] > highest_high[i] and close[i] > ema_50_aligned[i] and volume_confirm:
                # Bullish breakout above upper channel and above 1d EMA50
                signals[i] = 0.30
                position = 1
            elif close[i] < lowest_low[i] and close[i] < ema_50_aligned[i] and volume_confirm:
                # Bearish breakout below lower channel and below 1d EMA50
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price retraces to midpoint of Donchian channel or volume drops
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if close[i] < midpoint or volume[i] < vol_ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price retraces to midpoint of Donchian channel or volume drops
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if close[i] > midpoint or volume[i] < vol_ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals