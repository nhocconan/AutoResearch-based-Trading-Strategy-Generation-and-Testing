#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeConfirm
Hypothesis: Elder Ray (Bull/Bear Power) on 6h with 1d EMA50 trend filter and volume confirmation.
Bull Power = High - EMA13, Bear Power = Low - EMA13. Long when Bull Power > 0 and rising, 1d uptrend, volume > 1.5x average.
Short when Bear Power < 0 and falling, 1d downtrend, volume > 1.5x average.
Elder Ray measures price strength relative to EMA, effective in both trending and ranging markets.
Target: 12-25 trades/year on 6h timeframe to minimize fee drag while capturing strong directional moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h EMA13 for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 6h EMA13, 1d EMA50, volume MA
    start_idx = max(13, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 and rising (current > previous), 1d uptrend, volume confirmation
            bull_rising = bull_power[i] > bull_power[i-1]
            long_setup = (bull_power[i] > 0) and bull_rising and (close[i] > ema_50_1d_aligned[i]) and volume_confirm[i]
            
            # Short: Bear Power < 0 and falling (current < previous), 1d downtrend, volume confirmation
            bear_falling = bear_power[i] < bear_power[i-1]
            short_setup = (bear_power[i] < 0) and bear_falling and (close[i] < ema_50_1d_aligned[i]) and volume_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: Bull Power <= 0 OR 1d trend turns down OR volume drops
            if (bull_power[i] <= 0) or (close[i] < ema_50_1d_aligned[i]) or (not volume_confirm[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: Bear Power >= 0 OR 1d trend turns up OR volume drops
            if (bear_power[i] >= 0) or (close[i] > ema_50_1d_aligned[i]) or (not volume_confirm[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0