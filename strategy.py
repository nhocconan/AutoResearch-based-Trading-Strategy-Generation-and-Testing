#!/usr/bin/env python3
"""
4h_12h_Donchian_Breakout_Volume_Trend
Hypothesis: Use 4h Donchian(20) breakout with 12h EMA(34) trend filter and volume confirmation.
Long when price breaks above Donchian upper band with volume > 1.5x average and price > 12h EMA(34).
Short when price breaks below Donchian lower band with volume > 1.5x average and price < 12h EMA(34).
Exit when price returns to Donchian middle band (20-period average of high/low).
Designed for 4h timeframe to capture medium-term trends with ~20-40 trades/year.
Works in bull markets by buying breakouts above resistance and in bear markets by selling breakdowns below support.
Volume confirmation filters false breakouts, trend filter ensures alignment with higher timeframe momentum.
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
    
    # Calculate 12h EMA(34)
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Upper band: highest high of last 20 periods
    upper = np.full(n, np.nan)
    # Lower band: lowest low of last 20 periods  
    lower = np.full(n, np.nan)
    # Middle band: average of upper and lower
    middle = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or np.isnan(ema_34_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: break above upper band + volume + trend filter
            if price > upper[i] and volume_ok and price > ema_34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower band + volume + trend filter
            elif price < lower[i] and volume_ok and price < ema_34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle band
            if price < middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle band
            if price > middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0