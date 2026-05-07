#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_Volume_v2
Hypothesis: Price breaking above/below daily Camarilla R1/S1 with 1d EMA34 trend filter and volume spike (2x average) captures institutional breakouts. Tightened volume filter (2.0x) and added ATR(14) volatility filter to reduce false signals and lower trade frequency. Works in bull/bear by following higher timeframe trend. Target: 50-100 total trades over 4 years (12-25/year) to avoid fee drag.
"""
name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume_v2"
timeframe = "4h"
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
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (R1, S1) from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Pivot point
    pivot = (prev_high + prev_low + prev_close) / 3
    # Camarilla levels
    r1 = pivot + 1.1 * (prev_high - prev_low) / 12
    s1 = pivot - 1.1 * (prev_high - prev_low) / 12
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 2.0 * 24-period average (stricter)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    # ATR(14) filter: only trade when volatility is elevated
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[tr1[0]], tr2])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    vol_filter = atr > (pd.Series(atr).rolling(window=50, min_periods=50).mean().values * 0.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need sufficient warmup for ATR and averages
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + 1d uptrend + volume spike + vol filter
            if close[i] > r1_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_filter[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + 1d downtrend + volume spike + vol filter
            elif close[i] < s1_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_filter[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite Camarilla level (mean reversion)
            if position == 1:
                if close[i] <= s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals