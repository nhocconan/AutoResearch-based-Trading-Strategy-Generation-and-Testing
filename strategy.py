#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using Donchian channel breakout (20-period) with volume confirmation and ATR-based trailing stop.
- Long when price breaks above upper Donchian(20) + volume > 1.5x 20-period volume MA
- Short when price breaks below lower Donchian(20) + volume > 1.5x 20-period volume MA
- Fixed position size 0.25 to limit fee churn and manage drawdown
- ATR(10) trailing stop (2.0x ATR) to lock in profits
- Designed for low trade frequency (target: 20-50 trades per year) to avoid fee drag
- Works in bull markets (buying breakouts with volume) and bear markets (selling breakdowns with volume)
"""

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
    
    # Donchian channel (20-period) on primary 4h timeframe
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) on 4h for confirmation
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (10-period) on 4h for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(atr_10[i])):
            signals[i] = 0.0
            continue
        
        upper = highest_20[i]
        lower = lowest_20[i]
        vol_ma = volume_ma_20[i]
        atr_val = atr_10[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation
            # Long: price breaks above upper Donchian + volume spike
            if price > upper and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.0 * atr_val
            # Short: price breaks below lower Donchian + volume spike
            elif price < lower and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_stop = entry_price + 2.0 * atr_val
        
        elif position == 1:
            # Check stoploss
            if price <= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail stop: raise stop if price moves favorably
                atr_stop = max(atr_stop, price - 1.5 * atr_val)
        
        elif position == -1:
            # Check stoploss
            if price >= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stop: lower stop if price moves favorably
                atr_stop = min(atr_stop, price + 1.5 * atr_val)
    
    return signals

name = "4h_Donchian20_VolumeSpike_ATRTrail"
timeframe = "4h"
leverage = 1.0