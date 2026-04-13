#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation.
Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13.
Trend filter: 1d EMA50 > EMA200 for bullish regime, EMA50 < EMA200 for bearish regime.
Volume: 6h volume > 1.5x 20-period average.
Long in bullish regime when Bull Power > 0 and volume confirmation.
Short in bearish regime when Bear Power < 0 and volume confirmation.
Uses EMA13 for responsiveness. Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 and EMA200 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Bullish regime: EMA50 > EMA200, Bearish regime: EMA50 < EMA200
    bullish_regime = ema50_1d > ema200_1d
    bearish_regime = ema50_1d < ema200_1d
    
    bullish_regime_aligned = align_htf_to_ltf(prices, df_1d, bullish_regime.astype(float))
    bearish_regime_aligned = align_htf_to_ltf(prices, df_1d, bearish_regime.astype(float))
    
    # Calculate EMA13 for Elder Ray (6h)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # Volume confirmation (6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bullish_regime_aligned[i]) or 
            np.isnan(bearish_regime_aligned[i]) or 
            np.isnan(ema13[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = (bullish_regime_aligned[i] > 0.5 and 
                     bull_power[i] > 0 and 
                     vol_confirm[i])
        short_entry = (bearish_regime_aligned[i] > 0.5 and 
                      bear_power[i] < 0 and 
                      vol_confirm[i])
        
        # Exit when Elder Ray signal reverses
        exit_long = position == 1 and bull_power[i] <= 0
        exit_short = position == -1 and bear_power[i] >= 0
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_elder_ray_trend_volume"
timeframe = "6h"
leverage = 1.0