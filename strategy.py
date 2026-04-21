#!/usr/bin/env python3
"""
4h_12h_Donchian_Breakout_Volume_Trend
Hypothesis: Breakouts of 20-period Donchian channels on 4h timeframe, filtered by 12h EMA trend and volume confirmation.
Works in bull markets by buying breakouts above upper band and in bear markets by selling breakdowns below lower band.
Volume > 1.5x 20-period average confirms breakout strength.
Trend filter ensures trades align with higher timeframe momentum.
Designed for ~20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data once for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 12h close
    close_12h = df_12h['close'].values
    ema_34 = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if EMA not ready
        if np.isnan(ema_34_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        high = prices['high'].iloc[i]
        low = prices['low'].iloc[i]
        close = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Calculate Donchian channels (20-period)
        if i >= 20:
            high_window = prices['high'].iloc[i-20:i]
            low_window = prices['low'].iloc[i-20:i]
            upper_band = high_window.max()
            lower_band = low_window.min()
            
            # Volume filter: current volume > 1.5 * 20-period average
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            upper_band = lower_band = np.nan
            volume_ok = False
        
        if position == 0:
            # Long conditions: break above upper band + volume + uptrend (price > EMA)
            if not np.isnan(upper_band) and high > upper_band and volume_ok and close > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower band + volume + downtrend (price < EMA)
            elif not np.isnan(lower_band) and low < lower_band and volume_ok and close < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower band or trend turns down
            if not np.isnan(lower_band) and low < lower_band or close < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above upper band or trend turns up
            if not np.isnan(upper_band) and high > upper_band or close > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0