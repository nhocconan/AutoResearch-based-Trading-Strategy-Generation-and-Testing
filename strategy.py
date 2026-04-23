#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian breakout with 4h trend filter (HMA21) and volume spike confirmation.
- Donchian(20): Upper = highest high past 20 bars, Lower = lowest low past 20 bars
- 4h HMA21 for trend filter (bullish when price > HMA, bearish when price < HMA)
- Volume > 2.0x 20-period average for conviction
- Long: Close > Donchian Upper + volume confirmation + price > 4h HMA21
- Short: Close < Donchian Lower + volume confirmation + price < 4h HMA21
- Exit: Opposite Donchian breakout (Close < Upper for long, Close > Lower for short) or trend flip
- Uses 4h for signal direction (reduces noise), 1h only for entry timing precision
- Session filter: 08-20 UTC to avoid low-volume Asian session
- Discrete position sizing: ±0.20 to minimize fee churn
- Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe
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
    
    # Pre-compute hour filter for session (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1h Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h HMA21 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_4h).rolling(window=half_len, min_periods=half_len).mean().values
    wma_full = pd.Series(close_4h).rolling(window=21, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_21 = pd.Series(raw_hma).rolling(window=sqrt_len, min_periods=sqrt_len).mean().values
    hma_21_aligned = align_htf_to_ltf(prices, df_4h, hma_21)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 21)  # Need 20 for Donchian, 21 for HMA
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i] or \
           np.isnan(vol_ma[i]) or \
           np.isnan(donchian_upper[i]) or \
           np.isnan(donchian_lower[i]) or \
           np.isnan(hma_21_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close > Donchian Upper + volume confirmation + price > 4h HMA21
            if (close[i] > donchian_upper[i] and 
                volume_confirm and 
                close[i] > hma_21_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: Close < Donchian Lower + volume confirmation + price < 4h HMA21
            elif (close[i] < donchian_lower[i] and 
                  volume_confirm and 
                  close[i] < hma_21_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Close < Donchian Upper OR price < 4h HMA21 (trend flip)
            if close[i] < donchian_upper[i] or close[i] < hma_21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Close > Donchian Lower OR price > 4h HMA21 (trend flip)
            if close[i] > donchian_lower[i] or close[i] > hma_21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_DonchianBreakout_4hHMA21_VolumeSpike"
timeframe = "1h"
leverage = 1.0