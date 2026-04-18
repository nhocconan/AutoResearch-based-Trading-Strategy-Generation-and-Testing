#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_Plus
Hypothesis: Donchian(20) breakouts with volume confirmation and 4h EMA(34) trend filter capture strong directional moves.
Works in both bull and bear markets by following momentum with volatility-adjusted exits.
Target: 25-40 trades/year (100-160 total over 4 years) to balance opportunity and fee drag.
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
    
    # Donchian channels (20-period high/low)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: >1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.8 * vol_ma)
    
    # 4h EMA trend filter
    ema_fast = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_slow = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # ATR for volatility-based exit (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = 34  # Warmup for slow EMA
    
    for i in range(start_idx, n):
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(volume_filter[i]) or np.isnan(ema_fast[i]) or
            np.isnan(ema_slow[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_roll[i]
        lower = low_roll[i]
        vol_ok = volume_filter[i]
        ema_fast_val = ema_fast[i]
        ema_slow_val = ema_slow[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume in uptrend
            if price > upper and vol_ok and ema_fast_val > ema_slow_val:
                signals[i] = 0.30
                position = 1
                entry_price = price
                atr_at_entry = atr_val
            # Short: break below lower Donchian with volume in downtrend
            elif price < lower and vol_ok and ema_fast_val < ema_slow_val:
                signals[i] = -0.30
                position = -1
                entry_price = price
                atr_at_entry = atr_val
        
        elif position == 1:
            # Exit: price closes below lower Donchian OR trend reversal OR ATR-based stop
            if price < lower or ema_fast_val < ema_slow_val or price < entry_price - 2.5 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit: price closes above upper Donchian OR trend reversal OR ATR-based stop
            if price > upper or ema_fast_val > ema_slow_val or price > entry_price + 2.5 * atr_at_entry:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend_Plus"
timeframe = "4h"
leverage = 1.0