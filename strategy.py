#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
- Uses Donchian channel (20-period high/low) from prior completed 12h candles to identify breakout levels.
- Breakout above upper band or below lower band with volume > 2.0x 20-bar average signals strong momentum.
- Trend filter: price must be above/below 1w EMA50 to align with higher timeframe direction.
- Designed for 12h timeframe to capture medium-term breakouts in both bull and bear markets.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 12-37 trades/year (50-150 total over 4 years) to stay fee-efficient.
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
    
    # Prior completed 1w close for EMA50
    close_1w = df_1w['close'].shift(1).values
    
    # 1w EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to LTF
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Donchian breakout calculation (using prior 20 completed 12h bars)
            if i >= 20:
                highest_high = np.max(high[i-20:i])
                lowest_low = np.min(low[i-20:i])
                
                # Long: breakout above upper band AND price above 1w EMA50 AND volume confirmation
                if close[i] > highest_high and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                # Short: breakout below lower band AND price below 1w EMA50 AND volume confirmation
                elif close[i] < lowest_low and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: close below lower band OR price below 1w EMA50
            if i >= 20:
                lowest_low = np.min(low[i-20:i])
                if close[i] < lowest_low or close[i] < ema_50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above upper band OR price above 1w EMA50
            if i >= 20:
                highest_high = np.max(high[i-20:i])
                if close[i] > highest_high or close[i] > ema_50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1wEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0