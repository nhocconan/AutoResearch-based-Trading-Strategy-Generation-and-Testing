#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h, HTF: 12h for trend filter
- Long: Close breaks above Donchian upper (20-period high) + price > 12h EMA50 (uptrend) + volume > 1.8x 20-period avg
- Short: Close breaks below Donchian lower (20-period low) + price < 12h EMA50 (downtrend) + volume > 1.8x 20-period avg
- Exit: Close reverts to Donchian middle (10-period average of high/low) or ATR-based stoploss
- Uses Donchian breakouts with volume spike and HTF trend for controlled entries
- Target: 100-180 total trades over 4 years (25-45/year) on 4h timeframe
- Discrete position sizing: ±0.28 to balance return and risk
- BTC/ETH focus: requires HTF trend alignment to avoid SOL-only bias
- Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)
- Uses mtf_data helper for proper HTF alignment without look-ahead
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
    
    # Volume confirmation: > 1.8x 20-period average (volume spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_high + donchian_low) / 2.0
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # ATR for dynamic stoploss (optional, using signal=0 for exit)
    atr_period = 14
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for Donchian/volume, 50 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_middle[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.8x average)
        volume_spike = volume[i] > 1.8 * vol_ma[i]
        
        # ATR-based stoploss (2.5 * ATR)
        if not np.isnan(atr[i]) and atr[i] > 0:
            stop_loss_distance = 2.5 * atr[i]
        else:
            stop_loss_distance = np.inf  # Disable stop if ATR not ready
        
        if position == 0:
            # Long: Close breaks above Donchian upper + price > 12h EMA50 (uptrend) + volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike):
                signals[i] = 0.28
                position = 1
            # Short: Close breaks below Donchian lower + price < 12h EMA50 (downtrend) + volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike):
                signals[i] = -0.28
                position = -1
        elif position == 1:
            # Long exit: Close reverts to Donchian middle OR stoploss hit
            if (close[i] <= donchian_middle[i] or 
                (not np.isnan(atr[i]) and close[i] <= close[i-1] - stop_loss_distance)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # Short exit: Close reverts to Donchian middle OR stoploss hit
            if (close[i] >= donchian_middle[i] or 
                (not np.isnan(atr[i]) and close[i] >= close[i-1] + stop_loss_distance)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0