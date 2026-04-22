#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian breakout with 1-day trend filter.
Long when price breaks above Donchian(20) high and 1-day EMA(50) is rising.
Short when price breaks below Donchian(20) low and 1-day EMA(50) is falling.
Exit when price crosses Donchian midline or 1-day EMA trend reverses.
Uses price breakouts for momentum and EMA trend filter to avoid counter-trend trades.
Designed for 12h timeframe to capture multi-day trends while minimizing trade frequency.
Works in bull markets by catching breakouts and in bear markets by avoiding false breakouts during reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Load 1-day data for EMA trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_prev = np.roll(ema_50_1d, 1)
    ema_50_1d_prev[0] = ema_50_1d[0]
    ema_rising = ema_50_1d > ema_50_1d_prev
    ema_falling = ema_50_1d < ema_50_1d_prev
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high and 1-day EMA rising
            if close[i] > donch_high[i] and ema_rising_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low and 1-day EMA falling
            elif close[i] < donch_low[i] and ema_falling_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below Donchian midline OR EMA trend turns down
                if close[i] < donch_mid[i] or ema_rising_aligned[i] < 0.5:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above Donchian midline OR EMA trend turns up
                if close[i] > donch_mid[i] or ema_falling_aligned[i] < 0.5:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_1dEMA50_Trend"
timeframe = "12h"
leverage = 1.0