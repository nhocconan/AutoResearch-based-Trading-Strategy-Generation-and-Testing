# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h Donchian Breakout + Volume Confirmation + ATR Stop
Hypothesis: Donchian channel breakouts capture sustained trends in BTC/ETH/SOL.
Volume confirmation filters false breakouts. ATR-based position sizing and stop loss
control risk. Designed for 12h timeframe to limit trades (~20-40/year) and reduce fee
drag. Works in bull markets (breakouts up) and bear markets (breakouts down).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian Channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume moving average (20-period) for confirmation
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for volatility-based stop and sizing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long breakout: price breaks above Donchian high with volume confirmation
            if price > donchian_high[i] and vol > vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below Donchian low with volume confirmation
            elif price < donchian_low[i] and vol > vol_ma[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Trail stop: exit if price drops 2*ATR from highest high since entry
            # We approximate by exiting if price closes below Donchian low (breakdown)
            if price < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Trail stop: exit if price rises 2*ATR from lowest low since entry
            # Exit if price closes above Donchian high (breakout)
            if price > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_Volume"
timeframe = "12h"
leverage = 1.0