#!/usr/bin/env python3
"""
4h_1d_keltner_breakout_volume
Hypothesis: Uses 1-day Keltner Channel breakout with volume confirmation to capture strong momentum moves.
Works in bull markets (breakout continuation) and bear markets (mean reversion after sharp moves).
Targets 20-30 trades/year to minimize fee drag. Long and short positions for symmetry.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Keltner Channel (20, 2.0) on 1d
    typical_price = (high_1d + low_1d + close_1d) / 3
    atr_period = 20
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan
    
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            else:
                result[i] = np.nan
        return result
    
    atr = wilders_smooth(tr, atr_period)
    ema_atr = pd.Series(typical_price).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    upper_keltner = ema_atr + (2.0 * atr)
    lower_keltner = ema_atr - (2.0 * atr)
    
    # Breakout signals
    breakout_up = close_1d > upper_keltner
    breakout_down = close_1d < lower_keltner
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_expansion = df_1d['volume'].values > (vol_ma_20 * 1.5)
    
    # Align all signals to 4h timeframe
    breakout_up_aligned = align_htf_to_ltf(prices, df_1d, breakout_up.astype(float))
    breakout_down_aligned = align_htf_to_ltf(prices, df_1d, breakout_down.astype(float))
    volume_expansion_aligned = align_htf_to_ltf(prices, df_1d, volume_expansion.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(breakout_up_aligned[i]) or 
            np.isnan(breakout_down_aligned[i]) or 
            np.isnan(volume_expansion_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = breakout_up_aligned[i] > 0.5 and volume_expansion_aligned[i] > 0.5
        short_entry = breakout_down_aligned[i] > 0.5 and volume_expansion_aligned[i] > 0.5
        
        # Exit conditions: return to middle of Keltner Channel
        ema_aligned = align_htf_to_ltf(prices, df_1d, ema_atr)
        exit_long = position == 1 and close[i] >= ema_aligned[i]
        exit_short = position == -1 and close[i] <= ema_aligned[i]
        
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

name = "4h_1d_keltner_breakout_volume"
timeframe = "4h"
leverage = 1.0