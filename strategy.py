#!/usr/bin/env python3
"""
4h_EMA34_Trend_VolumeBreakout_1d
Hypothesis: Uses 1-day EMA34 for trend direction and 4-hour price action with volume confirmation for entries.
In uptrend (price > EMA34), go long on 4h breakout above 20-period high with volume spike.
In downtrend (price < EMA34), go short on 4h breakdown below 20-period low with volume spike.
Designed for low trade frequency by requiring trend alignment, breakout, and volume confirmation.
Works in both bull and bear markets by following the daily trend.
"""

name = "4h_EMA34_Trend_VolumeBreakout_1d"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1-day EMA34 Trend Filter ---
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # --- 4h 20-period High/Low for breakout ---
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(vol_ratio[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: uptrend + breakout above 20-period high + volume
            if (close[i] > ema_34_aligned[i] and 
                close[i] > high_20[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: downtrend + breakdown below 20-period low + volume
            elif (close[i] < ema_34_aligned[i] and 
                  close[i] < low_20[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or opposite breakout
            if position == 1:
                # Exit long: trend turns down OR breakdown below 20-period low
                if (close[i] < ema_34_aligned[i] or 
                    close[i] < low_20[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: trend turns up OR breakout above 20-period high
                if (close[i] > ema_34_aligned[i] or 
                    close[i] > high_20[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals