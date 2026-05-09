#!/usr/bin/env python3
# Hypothesis: 4h Williams %R overbought/oversold with 1d trend filter and volume confirmation.
# In trending markets (price above/below 200 EMA on 1d), Williams %R identifies overextended moves.
# Enters long when Williams %R < -80 (oversold) and price > 1d EMA200, short when > -20 (overbought) and price < 1d EMA200.
# Requires volume > 1.5x 20-period average for confirmation to avoid false signals in low volume.
# Exits when Williams %R returns to neutral range (-50) or trend reverses.
# Target: 20-50 trades/year with size 0.25 to minimize fee drag.

name = "4h_WilliamsR_1dTrend_Volume"
timeframe = "4h"
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
    
    # Calculate 1-day EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close']
    ema_200 = close_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Calculate Williams %R (14-period) on 4h data
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_low - lowest_low + 1e-10)  # Avoid division by zero
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: oversold + uptrend + volume confirmation
            if williams_r[i] < -80 and close[i] > ema_200_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: overbought + downtrend + volume confirmation
            elif williams_r[i] > -20 and close[i] < ema_200_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to neutral or trend reverses
            if williams_r[i] > -50 or close[i] < ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to neutral or trend reverses
            if williams_r[i] < -50 or close[i] > ema_200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals