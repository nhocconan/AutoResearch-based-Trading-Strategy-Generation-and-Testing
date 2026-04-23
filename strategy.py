#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
- Donchian(20) from prior completed weekly candle provides robust structure for breakouts.
- 1w EMA50 ensures alignment with weekly trend to avoid counter-trend whipsaws in bear markets.
- Volume > 2.0x 20-period average confirms breakout validity (tighter filter to reduce trades).
- Discrete position size 0.25 limits drawdown during crashes (BTC -77% → ~19% loss).
- Target: 10-25 trades/year on 1d timeframe (40-100 total over 4 years) to stay within fee-efficient range.
- Works in both bull and bear regimes via trend filter and volume confirmation.
- Uses HTF = 1w as specified in experiment #81458.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) from prior completed 1w (using mtf_data to avoid look-ahead)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Prior completed 1w high/low for Donchian channels
    high_1w = df_1w['high'].shift(1).values  # shift(1) to use prior completed week
    low_1w = df_1w['low'].shift(1).values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # 1w EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].shift(1)).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 2.0x 20-period average (tighter filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50, Donchian20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close > Donchian High AND price above 1w EMA50 AND volume confirmation
            if close[i] > donchian_high_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Close < Donchian Low AND price below 1w EMA50 AND volume confirmation
            elif close[i] < donchian_low_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < Donchian Low OR price crosses below 1w EMA50
            if close[i] < donchian_low_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > Donchian High OR price crosses above 1w EMA50
            if close[i] > donchian_high_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0