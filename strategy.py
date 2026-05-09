#!/usr/bin/env python3
# 4h_Camarilla_R1S1_Breakout_1dTrend
# Hypothesis: Breakouts from 1-day Camarilla R1/S1 levels with 1-day trend filter and volume confirmation.
# Works in bull markets (breakouts above R1 in uptrend) and bear markets (breakdowns below S1 in downtrend).
# Target: 20-35 trades/year per symbol with strict entry conditions to avoid overtrading.

name = "4h_Camarilla_R1S1_Breakout_1dTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA(34) for trend filter
    if len(close_1d) >= 34:
        ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    else:
        ema_34 = np.full_like(close_1d, np.nan)
    
    # Calculate daily Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    daily_range = high_1d - low_1d
    camarilla_R1 = close_1d + daily_range * 1.1 / 12
    camarilla_S1 = close_1d - daily_range * 1.1 / 12
    
    # Align daily indicators to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: 4h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or \
           np.isnan(ema_34_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above Camarilla R1 AND uptrend (close > EMA34) AND volume confirmation
            if close[i] > camarilla_R1_aligned[i] and close[i] > ema_34_aligned[i] and volume_ratio[i] > 2.0:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below Camarilla S1 AND downtrend (close < EMA34) AND volume confirmation
            elif close[i] < camarilla_S1_aligned[i] and close[i] < ema_34_aligned[i] and volume_ratio[i] > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price breaks below Camarilla S1
            if close[i] < camarilla_S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price breaks above Camarilla R1
            if close[i] > camarilla_R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals