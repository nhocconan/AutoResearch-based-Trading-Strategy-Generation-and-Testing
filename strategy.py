#!/usr/bin/env python3
"""
4h_1d_1w_Donchian20_Breakout_TrendVolume_Filtered_v1
Hypothesis: 4h Donchian(20) breakout in direction of 1d EMA50 trend, confirmed by volume spike.
Exit when price crosses opposite Donchian band or trend reverses.
Works in bull/bear by following daily trend and using volatility-based breakouts.
Target: 20-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h Donchian channels (20-period high/low)
    high = prices['high'].values
    low = prices['low'].values
    
    # Donchian upper (20-period high)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian lower (20-period low)
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long: price breaks above Donchian high, above daily EMA50, with volume
            if price > donch_high[i] and price > ema_50_1d_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, below daily EMA50, with volume
            elif price < donch_low[i] and price < ema_50_1d_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian low OR trend turns bearish
            if price < donch_low[i] or price < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian high OR trend turns bullish
            if price > donch_high[i] or price > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_1w_Donchian20_Breakout_TrendVolume_Filtered_v1"
timeframe = "4h"
leverage = 1.0