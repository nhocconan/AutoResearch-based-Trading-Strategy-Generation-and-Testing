#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_VolumeATRFilter_Tight
Hypothesis: Camarilla R1/S1 breakouts with volume confirmation and ATR stoploss on 4h timeframe, using 1d HTF for pivot calculation and 4h EMA34 for trend filter. Designed to work in both bull and bear markets by requiring confluence of price structure (breakout), momentum (volume), and regime (trend alignment). Target: 20-50 trades/year per symbol (80-200 over 4 years) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 4h data once for EMA34 trend filter (MTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla pivot point and R1/S1 levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12
    
    # Align daily levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4h EMA34 for trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume filter: 20-period average (approx 5 days on 4h)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_ok = volume > 1.5 * vol_ma[i]
        
        # 4h trend filter
        uptrend = close[i] > ema_34_4h_aligned[i]
        downtrend = close[i] < ema_34_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 in uptrend with volume
            if uptrend and volume_ok:
                if price > r1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below S1 in downtrend with volume
            elif downtrend and volume_ok:
                if price < s1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: price reaches S1 or ATR-based stoploss
            if price <= s1_aligned[i] or price < prices['close'].iloc[i-1] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price reaches R1 or ATR-based stoploss
            if price >= r1_aligned[i] or price > prices['close'].iloc[i-1] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_VolumeATRFilter_Tight"
timeframe = "4h"
leverage = 1.0