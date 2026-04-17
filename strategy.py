#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR trailing stop.
- Long when price breaks above 20-period Donchian high + volume > 1.5x 20-period volume MA
- Short when price breaks below 20-period Donchian low + volume > 1.5x 20-period volume MA
- Fixed position size 0.25 to limit fee churn and manage drawdown
- ATR(10) trailing stop (2.0x ATR) to lock in profits
- Works in bull markets (buying breakouts with volume) and bear markets (selling breakdowns with volume)
- Target: 75-200 trades over 4 years (19-50/year) to avoid fee drag
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
    
    # Donchian channel (20-period) on primary timeframe (4h)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (10-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 30  # warmup for Donchian(20) and ATR(10)
    
    for i in range(start_idx, n):
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or \
           np.isnan(volume_ma[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation
            # Long: price breaks above Donchian high + volume spike
            if price > donch_high[i] and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.0 * atr_val
            # Short: price breaks below Donchian low + volume spike
            elif price < donch_low[i] and vol > 1.5 * vol_ma:
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