#!/usr/bin/env python3
"""
6h Elder Ray Bull/Bear Power + 1d Williams Fractal Regime + Volume Confirmation
Hypothesis: Elder Ray measures bull/bear power via EMA(13) deviation. Williams Fractal on 1d identifies swing points for regime filtering. Trade in direction of daily trend (above/below pivot) with 6h Elder Ray signals confirmed by volume. Works in bull/bear by aligning with daily structure while using 6h momentum for timing.
Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # 1d Williams Fractals for regime (need 2-bar confirmation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Additional 2-bar delay for fractal confirmation (needs 2 future 1d bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # 1d pivot point (using previous day OHLC)
    pivot_1d = (df_1d['high'].shift(1) + df_1d['low'].shift(1) + df_1d['close'].shift(1)) / 3.0
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d.values)
    
    # 6h volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA13 + VolMA20 + HTF data
    start_idx = max(13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(pivot_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        pivot_level = pivot_1d_aligned[i]
        
        # Volume spike: current volume > 1.8 * 20-period average
        volume_spike = curr_volume > 1.8 * vol_ma_20[i]
        
        # Elder Ray conditions: bull power > 0 and rising, bear power < 0 and falling
        # Use 3-period momentum for power confirmation
        if i >= 3:
            bull_power_rising = bull_power[i] > bull_power[i-3]
            bear_power_falling = bear_power[i] < bear_power[i-3]
        else:
            bull_power_rising = False
            bear_power_falling = False
        
        bullish_momentum = bull_power[i] > 0 and bull_power_rising
        bearish_momentum = bear_power[i] < 0 and bear_power_falling
        
        # Regime filter: price above/below daily pivot + fractal confirmation
        bullish_regime = (curr_close > pivot_level) and (bullish_fractal_aligned[i] == 1.0)
        bearish_regime = (curr_close < pivot_level) and (bearish_fractal_aligned[i] == 1.0)
        
        # Exit conditions: opposite momentum or regime breakdown
        if position != 0:
            if position == 1 and (not bullish_momentum or not bullish_regime):
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and (not bearish_momentum or not bearish_regime):
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: momentum + regime + volume
        if position == 0:
            long_condition = bullish_momentum and bullish_regime and volume_spike
            short_condition = bearish_momentum and bearish_regime and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dWilliamsFractal_Regime_v1"
timeframe = "6h"
leverage = 1.0