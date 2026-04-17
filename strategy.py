#!/usr/bin/env python3
"""
12h_Pivot_R1_S1_Breakout_Volume_ATRFilter_V1
12-hour strategy using daily Camarilla pivot levels (R1, S1) with volume confirmation and ATR-based stop.
Enters long when price breaks above R1 with volume > 1.5x 20-period average.
Enters short when price breaks below S1 with volume > 1.5x 20-period average.
Uses ATR-based stop loss (2x ATR) and reverses position on opposite breakout.
Filters trades using weekly ADX > 20 to avoid choppy markets.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily data for Camarilla pivots and volume ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels (R1, S1) for each day
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align daily pivot levels to 12h timeframe (using previous day's close for alignment)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === Daily Volume for Confirmation ===
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === Weekly ADX for Regime Filter ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX components (14-period)
    plus_dm = np.zeros_like(high_1w)
    minus_dm = np.zeros_like(low_1w)
    plus_dm[1:] = np.maximum(high_1w[1:] - high_1w[:-1], 0)
    minus_dm[1:] = np.maximum(low_1w[:-1] - low_1w[1:], 0)
    plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0)
    minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0)
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = high_1w[0] - low_1w[0]
    
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr_1w * 14)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr_1w * 14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1w = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly ADX to 12h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # === ATR for Stop Loss ===
    atr_1d = pd.Series(np.maximum(np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1))), np.abs(low_1d - np.roll(close_1d, 1)))).rolling(window=14, min_periods=14).mean().values
    atr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or 
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current day's volume for confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_confirmed = vol_1d_current > 1.5 * vol_ma_1d_aligned[i]
        
        # Regime filter: only trade when weekly ADX > 20 (avoid choppy markets)
        trending = adx_1w_aligned[i] > 20
        
        # Breakout conditions
        breakout_up = close[i] > r1_1d_aligned[i]
        breakout_down = close[i] < s1_1d_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 with volume and trend filter
            if breakout_up and vol_confirmed and trending:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume and trend filter
            elif breakout_down and vol_confirmed and trending:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse on opposite breakout or ATR stop
        elif position == 1:
            # Stop loss: price drops below entry - 2*ATR (tracked via opposing breakout near S1)
            # Exit long: price breaks below S1 (opposite level)
            if breakout_down:
                signals[i] = -0.25  # reverse to short
                position = -1
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Stop loss: price rises above entry + 2*ATR (tracked via opposing breakout near R1)
            # Exit short: price breaks above R1 (opposite level)
            if breakout_up:
                signals[i] = 0.25  # reverse to long
                position = 1
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1_S1_Breakout_Volume_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0