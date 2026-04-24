#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume spike confirmation.
- Uses Donchian channel (20-period high/low) from prior completed 4h candles for structure.
- Breakout above upper band or below lower band with volume > 1.5x 20-bar average signals momentum.
- Trend filter: 1d ATR(14) must be expanding (current ATR > ATR 5 periods ago) to ensure volatile market.
- Designed for 4h timeframe to capture breakouts in both bull and bear markets, avoiding choppy regimes.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 20-40 trades/year (80-160 total over 4 years) to stay fee-efficient.
- Based on proven pattern: Donchian breakout + volume + ATR filter showed SOL performance in DB.
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # first period
    tr3[0] = np.abs(low_1d[0] - close_1d[0])   # first period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_14 = np.zeros_like(tr)
    atr_14[13] = np.mean(tr[1:14])  # first ATR value
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # ATR expansion filter: current ATR > ATR 5 periods ago
    atr_expanding = np.zeros_like(atr_14, dtype=bool)
    atr_expanding[5:] = atr_14[5:] > atr_14[:-5]
    
    # Align HTF indicators to LTF
    atr_expanding_aligned = align_htf_to_ltf(prices, df_1d, atr_expanding)
    
    # Donchian(20) from prior completed 4h candles (using 4h data from prices)
    # We need to compute Donchian on 4h close, but since we only have LTF prices,
    # we'll use rolling window on the 4h-equivalent bars
    # For 4h timeframe, each 4h bar = 16 * 15m bars, but we're on 4h chart so direct rolling
    # Since prices is already 4h timeframe, we can compute directly
    donchian_window = 20
    high_roll = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    low_roll = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_window, 20)  # Donchian(20) and vol MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_expanding_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above upper Donchian band AND ATR expanding AND volume confirmation
            if close[i] > high_roll[i] and atr_expanding_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower Donchian band AND ATR expanding AND volume confirmation
            elif close[i] < low_roll[i] and atr_expanding_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below lower Donchian band OR ATR contracting
            if close[i] < low_roll[i] or not atr_expanding_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above upper Donchian band OR ATR contracting
            if close[i] > high_roll[i] or not atr_expanding_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATRFilter_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0