#!/usr/bin/env python3
"""
exp_6456_12h_donchian20_1d_ema_vol_v1
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Works in bull/bear by requiring volume > 1.5x 20-bar average and EMA50 alignment.
Target: 75-150 total trades over 4 years.
"""

name = "exp_6456_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Precompute indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index 20 to ensure Donchian is valid
    start_idx = max(20, 50)  # Ensure EMA and Donchian are ready
    
    for i in range(start_idx, n):
        # Get current values
        dc_high = highest_high[i-1]  # Previous bar's Donchian high (breakout of prior 20)
        dc_low = lowest_low[i-1]     # Previous bar's Donchian low
        ema_trend = ema_1d_50_aligned[i]
        vol_ok = vol_filter[i]
        
        # Long condition: price breaks above Donchian high + EMA uptrend + volume
        if position == 0 and close[i] > dc_high and ema_trend > close_1d[i//16 if i//16 < len(close_1d) else -1] and vol_ok:
            signals[i] = 0.30
            position = 1
            entry_price = close[i]
        
        # Short condition: price breaks below Donchian low + EMA downtrend + volume
        elif position == 0 and close[i] < dc_low and ema_trend < close_1d[i//16 if i//16 < len(close_1d) else -1] and vol_ok:
            signals[i] = -0.30
            position = -1
            entry_price = close[i]
        
        # Exit conditions
        elif position == 1:
            # Stoploss: 2.5 * ATR-based (using 20-bar range as proxy)
            atr_proxy = (highest_high[i] - lowest_low[i]) / 2.0
            if close[i] < entry_price - 2.5 * atr_proxy:
                signals[i] = 0.0
                position = 0
            # Take profit: exit at opposite Donchian level
            elif close[i] < dc_low:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Stoploss
            atr_proxy = (highest_high[i] - lowest_low[i]) / 2.0
            if close[i] > entry_price + 2.5 * atr_proxy:
                signals[i] = 0.0
                position = 0
            # Take profit
            elif close[i] > dc_high:
                signals[i] = 0.0
                position = 0
    
    return signals