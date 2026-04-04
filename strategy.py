#!/usr/bin/env python3
"""
exp_6456_12h_donchian20_1d_ema_vol_v1
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Works in bull/bear: EMA50 filter ensures we only trade in the direction of the higher timeframe trend,
while Donchian breakout captures momentum. Volume confirmation reduces false signals.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6456_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if df_1d is None or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate HTF indicators
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from sufficient lookback
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Get aligned HTF values
        ema_1d_val = ema_1d_aligned[i]
        
        # Skip if HTF data not ready
        if np.isnan(ema_1d_val):
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # Long condition: price breaks above Donchian high + above HTF EMA + volume
        if position == 0:
            if (close[i] > donch_high[i] and 
                close[i] > ema_1d_val and 
                vol_ok):
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
        
        # Short condition: price breaks below Donchian low + below HTF EMA + volume
        elif position == 0:
            if (close[i] < donch_low[i] and 
                close[i] < ema_1d_val and 
                vol_ok):
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
        
        # Exit conditions
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry
            # Take profit: signal reduces to 0.15 at 2R profit
            if close[i] < entry_price - 2.5 * 0.01 * entry_price:  # Simplified ATR proxy
                signals[i] = 0.0
                position = 0
            elif close[i] > entry_price + 2.5 * 0.01 * entry_price:  # 2.5R profit
                signals[i] = 0.15  # Take half profit
        
        elif position == -1:  # Short position
            if close[i] > entry_price + 2.5 * 0.01 * entry_price:
                signals[i] = 0.0
                position = 0
            elif close[i] < entry_price - 2.5 * 0.01 * entry_price:
                signals[i] = -0.15  # Take half profit
    
    return signals