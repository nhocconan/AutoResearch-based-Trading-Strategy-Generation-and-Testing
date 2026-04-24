#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation (>2.0x 24-bar average).
- Uses discrete position size 0.30 to balance drawdown and fee efficiency.
- Targets 15-30 trades/year (60-120 total over 4 years) to stay fee-efficient.
- Combines Donchian structure + 1w trend filter + volume confirmation.
- Works in bull/bear: trend filter ensures alignment with higher timeframe direction; volume filter avoids low-conviction entries.
- Donchian breakouts capture momentum; 1w EMA50 filters for primary trend; volume confirmation ensures conviction.
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
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Prior 1w OHLC (completed 1w bar)
    high_1w = df_1w['high'].shift(1).values
    low_1w = df_1w['low'].shift(1).values
    close_1w = df_1w['close'].shift(1).values
    
    # Align to 12h timeframe
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1w EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 24)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close > Donchian High AND price above 1w EMA50 AND volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.30
                position = 1
            # Short: Close < Donchian Low AND price below 1w EMA50 AND volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: Close < Donchian Low OR price crosses below 1w EMA50
            if close[i] < donchian_low[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: Close > Donchian High OR price crosses above 1w EMA50
            if close[i] > donchian_high[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_Donchian20_Breakout_1wEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0