#!/usr/bin/env python3
"""
Hypothesis: Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation.
Goes long when price breaks above Donchian upper channel AND price > 1d EMA50 AND volume > 1.5x average volume.
Goes short when price breaks below Donchian lower channel AND price < 1d EMA50 AND volume > 1.5x average volume.
Uses ATR(14) stoploss (signal → 0 when price moves 2*ATR against position).
Position size: 0.25 (25% of capital).
Designed to work in both bull and bear markets by requiring volume confirmation and trend alignment.
Target: 75-200 total trades over 4 years (~19-50/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6449_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    # Calculate EMA(50) on 1d close
    close_1d = pd.Series(df_1d['close'].values)
    ema_1d_50 = close_1d.ewm(span=50, min_periods=50, adjust=False).mean().values
    # Align EMA to LTF (4h) with shift(1) to avoid look-ahead
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_roll = prices['high'].rolling(window=20, min_periods=20).max()
    low_roll = prices['low'].rolling(window=20, min_periods=20).min()
    donchian_upper = high_roll.values
    donchian_lower = low_roll.values
    
    # Calculate average volume for confirmation
    avg_volume = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) for stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First period has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position_side = 0  # 1 for long, -1 for short, 0 for flat
    entry_price = 0.0
    
    # Start from index 20 to ensure Donchian channels are calculated
    for i in range(20, n):
        # Check stoploss for existing position
        if position_side == 1:  # Long position
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position_side = 0
                continue
        elif position_side == -1:  # Short position
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position_side = 0
                continue
        
        # Skip if we don't have enough data for indicators
        if np.isnan(ema_1d_50_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(avg_volume[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = prices['volume'].iloc[i] > 1.5 * avg_volume[i]
        
        # Long entry: price breaks above Donchian upper AND price > 1d EMA50 AND volume confirmed
        if position_side == 0 and close[i] > donchian_upper[i] and close[i] > ema_1d_50_aligned[i] and volume_confirmed:
            signals[i] = 0.25  # Long 25% of capital
            position_side = 1
            entry_price = close[i]
        # Short entry: price breaks below Donchian lower AND price < 1d EMA50 AND volume confirmed
        elif position_side == 0 and close[i] < donchian_lower[i] and close[i] < ema_1d_50_aligned[i] and volume_confirmed:
            signals[i] = -0.25  # Short 25% of capital
            position_side = -1
            entry_price = close[i]
        # Exit flat if no position and no entry signal
        elif position_side == 0:
            signals[i] = 0.0
        # Maintain current position if no stoploss triggered
        else:
            signals[i] = 0.25 if position_side == 1 else -0.25
    
    return signals