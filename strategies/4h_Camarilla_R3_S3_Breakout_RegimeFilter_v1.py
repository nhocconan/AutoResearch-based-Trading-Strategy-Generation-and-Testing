#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_RegimeFilter_v1
Hypothesis: Camarilla R3/S3 breakouts with 1d EMA34 trend filter, volume spike confirmation, and choppiness regime filter (CHOP > 50) produce fewer, higher-quality trades by avoiding whipsaws in sideways markets. Works in both bull and bear via trend filter. Target: 80-120 total trades over 4 years (20-30/year).
"""

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
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_R3 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_S3 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe (completed 1d bars only)
    R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # 4h volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h choppiness index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14) over 14 periods) / (log10(highest high - lowest low over 14 periods))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_14 - lowest_low_14
    # Avoid division by zero
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10(sum_atr_14 / chop_denom) / np.log10(14)
    chop[np.isnan(chop)] = 100  # default to choppy when not enough data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA, 14 for CHOP)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Regime filter: only trade when market is not too choppy (CHOP <= 50)
        # CHOP > 50 indicates ranging/choppy market, avoid breakouts
        not_choppy = chop[i] <= 50
        
        # Volume spike condition
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Camarilla R3/S3 breakout conditions
        breakout_up = close[i] > R3_aligned[i]   # Price breaks above R3
        breakout_down = close[i] < S3_aligned[i]  # Price breaks below S3
        
        # 1d EMA34 trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if breakout_up and uptrend and volume_spike and not_choppy:
            # Long signal: break above R3 + uptrend + volume spike + not choppy
            if position != 1:
                signals[i] = 0.30
                position = 1
            else:
                signals[i] = 0.30
        elif breakout_down and downtrend and volume_spike and not_choppy:
            # Short signal: break below S3 + downtrend + volume spike + not choppy
            if position != -1:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = -0.30
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_RegimeFilter_v1"
timeframe = "4h"
leverage = 1.0