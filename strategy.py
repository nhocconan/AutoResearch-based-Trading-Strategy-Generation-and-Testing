#!/usr/bin/env python3
"""
6h_Structure_Trend_With_Pullback_Filter
Hypothesis: Trade in direction of 12h structure (higher highs/lows for long, lower for short) with 6h pullback to EMA21.
Works in bull/bear: structure filters countertrend moves, pullback entries reduce slippage and improve risk/reward.
Volume confirmation ensures institutional participation. Target: 60-120 total trades over 4 years (15-30/year).
"""

name = "6h_Structure_Trend_With_Pullback_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for structure determination
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h structure: higher highs and higher lows for uptrend
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Track swing points
    swing_high = np.full_like(high_12h, np.nan)
    swing_low = np.full_like(low_12h, np.nan)
    
    # Find swing highs and lows (3-bar lookback/forward)
    for i in range(3, len(high_12h) - 3):
        if high_12h[i] == np.max(high_12h[i-3:i+4]):
            swing_high[i] = high_12h[i]
        if low_12h[i] == np.min(low_12h[i-3:i+4]):
            swing_low[i] = low_12h[i]
    
    # Determine structure state: 1=uptrend (HH and HL), -1=downtrend (LH and LL), 0=unclear
    structure = np.zeros_like(high_12h)
    last_swing_high = np.nan
    last_swing_low = np.nan
    
    for i in range(len(high_12h)):
        if not np.isnan(swing_high[i]):
            last_swing_high = swing_high[i]
        if not np.isnan(swing_low[i]):
            last_swing_low = swing_low[i]
        
        if not np.isnan(last_swing_high) and not np.isnan(last_swing_low):
            if len([x for x in swing_high[:i+1] if not np.isnan(x)]) >= 2 and \
               len([x for x in swing_low[:i+1] if not np.isnan(x)]) >= 2:
                # Get last two swing points
                recent_highs = [x for x in swing_high[:i+1] if not np.isnan(x)][-2:]
                recent_lows = [x for x in swing_low[:i+1] if not np.isnan(x)][-2:]
                if len(recent_highs) == 2 and len(recent_lows) == 2:
                    if recent_highs[1] > recent_highs[0] and recent_lows[1] > recent_lows[0]:
                        structure[i] = 1  # uptrend
                    elif recent_highs[1] < recent_highs[0] and recent_lows[1] < recent_lows[0]:
                        structure[i] = -1  # downtrend
    
    # Align structure to 6h timeframe
    structure_aligned = align_htf_to_ltf(prices, df_12h, structure)
    
    # Calculate 6h EMA21 for pullback entries
    close = prices['close'].values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate volume confirmation (20-period average)
    volume = prices['volume'].values
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):  # Start after EMA21 warmup
        struct = structure_aligned[i]
        current_close = close[i]
        current_ema21 = ema21[i]
        current_volume = volume[i]
        current_vol_avg = vol_avg[i]
        
        # Volume spike: current volume > 1.3x average
        vol_confirm = current_volume > 1.3 * current_vol_avg
        
        if position == 0:
            # Long: uptrend structure + pullback to EMA21 + volume confirmation
            if struct == 1 and current_close >= current_ema21 * 0.995 and current_close <= current_ema21 * 1.005 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: downtrend structure + pullback to EMA21 + volume confirmation
            elif struct == -1 and current_close <= current_ema21 * 1.005 and current_close >= current_ema21 * 0.995 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: structure breaks down or price extends too far from EMA
            if struct == -1 or current_close > current_ema21 * 1.03:  # 3% extension
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: structure breaks up or price extends too far from EMA
            if struct == 1 or current_close < current_ema21 * 0.97:  # 3% extension
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals