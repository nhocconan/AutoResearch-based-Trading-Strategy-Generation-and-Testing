#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Breakout_v1
Hypothesis: Uses Camarilla pivot levels from daily timeframe with volume confirmation
and ADX trend filter to identify high-probability breakouts. Works in both bull and bear
markets by capturing volatility expansions after consolidation. Targets 20-40 trades/year.
"""

name = "4h_Camarilla_Pivot_Breakout_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Daily Camarilla pivot levels (based on prior day's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels: based on previous day's high-low-close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid division by zero in range calculation
    daily_range = prev_high - prev_low
    daily_range = np.where(daily_range == 0, 1e-10, daily_range)
    
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    camarilla_pp = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 6
    camarilla_r2 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_r4 = prev_close + (prev_high - prev_low) * 1.1
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 6
    camarilla_s2 = prev_close - (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    camarilla_s4 = prev_close - (prev_high - prev_low) * 1.1
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels for current day)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX trend filter (14-period) - use only when trending (ADX > 25)
    # Calculate directional movement
    high_diff = np.diff(high, prepend=high[0])
    low_diff = np.diff(low, prepend=low[0])
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # True range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first bar
    tr2[0] = np.abs(high[0] - close[0])  # first bar
    tr3[0] = np.abs(low[0] - close[0])  # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth TR, +DM, -DM
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    plus_di = np.divide(plus_dm_sum, tr_sum, out=np.zeros_like(tr_sum), where=tr_sum!=0) * 100
    minus_di = np.divide(minus_dm_sum, tr_sum, out=np.zeros_like(tr_sum), where=tr_sum!=0) * 100
    dx = np.divide(np.abs(plus_di - minus_di), (plus_di + minus_di), out=np.zeros_like(tr_sum), where=(plus_di + minus_di)!=0) * 100
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(28, n):  # Start after ADX warmup
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_r2_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(camarilla_s2_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(adx[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade when ADX indicates trending market (ADX > 25)
        is_trending = adx[i] > 25
        
        if position == 0 and is_trending:
            # Long breakout: price breaks above R1 with volume confirmation
            if close[i] > camarilla_r1_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume confirmation
            elif close[i] < camarilla_s1_aligned[i] and volume[i] > vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 (reversal) or loses momentum (ADX < 20)
            if close[i] < camarilla_s1_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 (reversal) or loses momentum (ADX < 20)
            if close[i] > camarilla_r1_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals