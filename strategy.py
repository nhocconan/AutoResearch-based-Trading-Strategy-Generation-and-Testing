#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_ATR_Volume_Regime
Hypothesis: 4h strategy using Donchian(20) breakouts with ATR-based stoploss and volume confirmation. 
Donchian channels provide clear structure for breakouts in both trending and ranging markets. 
ATR stoploss adapts to volatility, limiting drawdowns during crashes like 2022. 
Volume confirmation ensures breakouts have institutional participation, reducing false signals. 
Designed for BTC/ETH robustness with discrete position sizing (0.25) to minimize fee drag. 
Targets 75-200 trades over 4 years (19-50/year) with strong risk control.
"""

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
    
    # Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR(14) for volatility-based stoploss
    atr_period = 14
    tr1 = pd.Series(high).rolling(window=2).max() - pd.Series(low).rolling(window=2).min()
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Fixed position size to minimize churn
    
    # Warmup: need Donchian(20), ATR(14), vol avg(20)
    start_idx = max(lookback, atr_period, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(atr[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        atr_val = atr[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with volume confirmation
            long_condition = (close_val > upper and vol_conf)
            short_condition = (close_val < lower and vol_conf)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower OR 2*ATR stoploss
            if close_val < lower or close_val < (high_val - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above Donchian upper OR 2*ATR stoploss
            if close_val > upper or close_val > (low_val + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_ATR_Volume_Regime"
timeframe = "4h"
leverage = 1.0