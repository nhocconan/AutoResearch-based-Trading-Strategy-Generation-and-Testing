#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 Breakout with 1d EMA34 trend filter and volume spike.
Long when price breaks above R3 (1d pivot resistance) AND 1d EMA34 uptrend AND volume > 2.0x 20-period average.
Short when price breaks below S3 (1d pivot support) AND 1d EMA34 downtrend AND volume > 2.0x 20-period average.
Exit when price reverts to 1d EMA34 or opposite Camarilla level (S3 for longs, R3 for shorts).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-40 trades/year per symbol.
Camarilla levels provide precise intraday support/resistance derived from prior day's range.
1d EMA34 ensures alignment with higher-timeframe momentum. Volume confirmation filters weak breakouts.
Designed to work in both bull and bear markets by requiring HTF trend alignment and volatility expansion.
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
    
    # Load 1d data for Camarilla levels and EMA34 - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate previous day's Camarilla levels for current day
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use prior completed 1d bar to avoid look-ahead
    shift_high = np.roll(high_1d, 1)
    shift_low = np.roll(low_1d, 1)
    shift_close = np.roll(close_1d, 1)
    shift_high[0] = np.nan
    shift_low[0] = np.nan
    shift_close[0] = np.nan
    
    rang = shift_high - shift_low
    R3 = shift_close + 1.1 * rang
    S3 = shift_close - 1.1 * rang
    
    # Align HTF indicators to 4h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 34)  # Ensure warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 AND 1d EMA34 uptrend AND volume spike
            if (price > R3_aligned[i] and 
                ema34_aligned[i] > ema34_aligned[i-1] and  # EMA34 rising
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND 1d EMA34 downtrend AND volume spike
            elif (price < S3_aligned[i] and 
                  ema34_aligned[i] < ema34_aligned[i-1] and  # EMA34 falling
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price reverts to 1d EMA34
            if position == 1 and price <= ema34_aligned[i]:
                exit_signal = True
            elif position == -1 and price >= ema34_aligned[i]:
                exit_signal = True
            
            # Alternative exit: price reaches opposite Camarilla level
            elif position == 1 and price < S3_aligned[i]:
                exit_signal = True
            elif position == -1 and price > R3_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0