#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using Donchian(20) breakout with volume confirmation and ATR trailing stop.
- Long when price breaks above Donchian upper(20) + volume > 1.5x 20-period volume MA
- Short when price breaks below Donchian lower(20) + volume > 1.5x 20-period volume MA
- Fixed position size 0.25 to limit fee churn and manage drawdown
- ATR(10) trailing stop (2.0x ATR) to lock in profits
- Designed for moderate trade frequency (target: 75-200 trades over 4 years) to avoid fee drag
- Works in bull markets (buying breakouts) and bear markets (selling breakdowns)
- Added choppiness regime filter: only trade when CHOP(14) < 61.8 (trending market)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on primary timeframe
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) for confirmation
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (10-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(highest_high - lowest_low))) / log10(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_14 - lowest_low_14
    # Avoid division by zero
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10(sum_atr_14 / chop_denom) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(atr_10[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20[i]
        atr_val = atr_10[i]
        upper = highest_20[i]
        lower = lowest_20[i]
        chop_val = chop[i]
        
        # Only trade in trending markets (CHOP < 61.8)
        if chop_val >= 61.8:
            # In choppy/ranging markets, stay flat
            if position == 1:
                signals[i] = 0.0
                position = 0
            elif position == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for breakouts with volume confirmation
            # Long: price breaks above Donchian upper + volume spike
            if price > upper and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.0 * atr_val
            # Short: price breaks below Donchian lower + volume spike
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

name = "4h_Donchian20_VolumeSpike_ChopFilter_ATRTrail"
timeframe = "4h"
leverage = 1.0