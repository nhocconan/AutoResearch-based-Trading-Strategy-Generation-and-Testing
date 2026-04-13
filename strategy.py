#!/usr/bin/env python3
"""
4h_1d_KC_Donchian_Breakout
Hypothesis: Combining Keltner Channel mean-reversion with Donchian breakout filters false signals.
In ranging markets, price tends to revert to the Keltner middle (EMA); in trending markets,
breakouts of Donchian channels with volume confirmation capture moves. The 1d trend filter
(EMA50) ensures alignment with higher timeframe direction, working in both bull and bear regimes.
Target: 25-40 trades/year.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Keltner Channel (20, 2.0)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(high - low).rolling(window=20, min_periods=20).mean().values
    kc_upper = ema_20 + 2.0 * atr
    kc_lower = ema_20 - 2.0 * atr
    
    # Donchian Channel (20)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(kc_upper[i]) or 
            np.isnan(kc_lower[i]) or np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price below Keltner lower (oversold) AND
        # 2. Breakout above Donchian high with volume expansion AND
        # 3. Above 1d EMA50 (uptrend filter)
        oversold = close[i] < kc_lower[i]
        donch_breakout = close[i] > donch_high[i]
        long_condition = oversold and donch_breakout and volume_expansion[i] and (close[i] > ema_50_1d_aligned[i])
        
        # Short conditions:
        # 1. Price above Keltner upper (overbought) AND
        # 2. Breakdown below Donchian low with volume expansion AND
        # 3. Below 1d EMA50 (downtrend filter)
        overbought = close[i] > kc_upper[i]
        donch_breakdown = close[i] < donch_low[i]
        short_condition = overbought and donch_breakdown and volume_expansion[i] and (close[i] < ema_50_1d_aligned[i])
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_KC_Donchian_Breakout"
timeframe = "4h"
leverage = 1.0