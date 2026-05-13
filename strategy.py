#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d HMA34 trend filter and volume confirmation (>2.0x 20-bar avg volume).
# Uses Donchian channel for structure, HMA34 for smooth 1d trend alignment, and high volume threshold to reduce false signals.
# Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drag while capturing strong momentum moves.
# Exit on reverse Donchian break or volume drop below 50% of average. Works in bull/bear via trend-following breakouts.

name = "12h_Donchian20_1dHMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d HMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    half = 34 // 2
    sqrt_n = int(np.sqrt(34))
    wma_half = pd.Series(close_1d).ewm(span=half, adjust=False).mean().values
    wma_full = pd.Series(close_1d).ewm(span=34, adjust=False).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_34_1d = pd.Series(raw_hma).ewm(span=sqrt_n, adjust=False).mean().values
    hma_34_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_34_1d)
    
    # Calculate 1d Donchian channels (20-period) based on prior 1d bar
    lookback_donch = 20
    if len(df_1d) < lookback_donch:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Rolling max/min for prior 20 1d bars
    donch_high = pd.Series(high_1d).rolling(window=lookback_donch, min_periods=lookback_donch).max().shift(1).values
    donch_low = pd.Series(low_1d).rolling(window=lookback_donch, min_periods=lookback_donch).min().shift(1).values
    
    # Align Donchian levels to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_donch, lookback_vol) + 1, n):
        # Skip if any required data is NaN
        if (np.isnan(hma_34_1d_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 1d Donchian high, close > 1d HMA34, volume spike (>2.0x avg)
            if (high[i] > donch_high_aligned[i] and 
                close[i] > hma_34_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 1d Donchian low, close < 1d HMA34, volume spike (>2.0x avg)
            elif (low[i] < donch_low_aligned[i] and 
                  close[i] < hma_34_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close if price breaks below 1d Donchian low or volume drops below 50% avg
            if (low[i] < donch_low_aligned[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close if price breaks above 1d Donchian high or volume drops below 50% avg
            if (high[i] > donch_high_aligned[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals