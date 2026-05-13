#!/usr/bin/env python3
# Hypothesis: 6h Weekly Donchian(20) breakout with daily volume confirmation and ATR filter.
# Long when price breaks above weekly Donchian high AND daily volume > 1.5x daily volume MA20 AND ATR(14) < 0.03 * close (low volatility regime).
# Short when price breaks below weekly Donchian low AND daily volume > 1.5x daily volume MA20 AND ATR(14) < 0.03 * close.
# Uses weekly structure for direction, daily volume for confirmation, and ATR filter to avoid high-volatility whipsaws.
# Designed to capture sustained moves with tight entries (target: 12-30 trades/year) and avoid overtrading.

name = "6h_WeeklyDonchian20_Volume_ATRFilter_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian(20) channels
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to 6h timeframe (wait for completed weekly bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate daily volume MA20
    volume_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily volume MA20 to 6h timeframe
    volume_ma20_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_ma20_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly Donchian high AND volume > 1.5x daily MA20 AND low volatility
            if (close[i] > donchian_high_aligned[i] and 
                volume[i] > 1.5 * volume_ma20_aligned[i] and 
                atr[i] < 0.03 * close[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly Donchian low AND volume > 1.5x daily MA20 AND low volatility
            elif (close[i] < donchian_low_aligned[i] and 
                  volume[i] > 1.5 * volume_ma20_aligned[i] and 
                  atr[i] < 0.03 * close[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below weekly Donchian low (reversal signal)
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly Donchian high (reversal signal)
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals