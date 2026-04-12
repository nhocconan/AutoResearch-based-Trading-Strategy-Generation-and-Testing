#!/usr/bin/env python3
"""
12h_1d_donchian_volatility_breakout
Uses Donchian channel breakout on 12h with volatility filter and 1d trend filter.
Enters long when price breaks above upper Donchian band with volatility expansion and 1d uptrend.
Enters short when price breaks below lower Donchian band with volatility expansion and 1d downtrend.
Exits when price returns to middle of Donchian channel.
Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drag.
Works in both trending and volatile markets by combining breakout with volatility confirmation.
"""

name = "12h_1d_donchian_volatility_breakout"
timeframe = "12h"
leverage = 1.0

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
    
    # Donchian channel parameters
    donchian_period = 20
    
    # Calculate Donchian bands
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    middle = (highest_high + lowest_low) / 2
    
    # Volatility filter: ATR expansion
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_expansion = atr > (atr_ma * 1.2)  # ATR above 20-period average
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 50-period EMA for daily trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above upper Donchian band with volatility expansion and daily uptrend
        if (close[i] > highest_high[i] and vol_expansion[i] and 
            close[i] > ema_50_1d_aligned[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price breaks below lower Donchian band with volatility expansion and daily downtrend
        elif (close[i] < lowest_low[i] and vol_expansion[i] and 
              close[i] < ema_50_1d_aligned[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price returns to middle of Donchian channel
        elif position == 1 and close[i] < middle[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > middle[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals