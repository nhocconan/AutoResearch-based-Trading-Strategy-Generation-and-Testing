#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot R1/S1 breakout + volume confirmation + 1d ADX trend filter.
Long when price breaks above Camarilla R1 AND volume > 1.3x average AND 1d ADX > 25.
Short when price breaks below Camarilla S1 AND volume > 1.3x average AND 1d ADX > 25.
Exit when price reverts to Camarilla midpoint (close of prior day) OR 1d ADX < 20.
Uses 4h for price action/volume and 1d for ADX regime filter to avoid whipsaw in ranging markets.
Target: 75-200 total trades over 4 years (19-50/year). Camarilla levels provide precise intraday
support/resistance derived from prior day's range, effective in both trending and mean-reverting
markets when combined with trend filter. Volume confirmation reduces false breakouts.
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
    
    # Get 4h data for price/volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Camarilla levels on 4h using prior 1d OHLC (already in df_4h as it's 4h data)
    # Camarilla requires prior day's OHLC - we'll get 1d data for this
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on prior day's range
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    # Midpoint = close (prior day close)
    range_1d = high_1d - low_1d
    camarilla_r1 = close_1d + range_1d * 1.1 / 12
    camarilla_s1 = close_1d - range_1d * 1.1 / 12
    camarilla_mid = close_1d  # midpoint is prior day close
    
    # Calculate ADX on 1d timeframe (14-period)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where(np.isnan(dx) | np.isinf(dx), 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume average (20-period) on 4h
    volume_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        mid = camarilla_mid_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Camarilla R1 AND volume > 1.3x avg AND ADX > 25 (trending)
            if price > r1 and vol > 1.3 * vol_ma and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price < Camarilla S1 AND volume > 1.3x avg AND ADX > 25 (trending)
            elif price < s1 and vol > 1.3 * vol_ma and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Camarilla midpoint OR ADX < 20 (losing trend)
            if price < mid or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Camarilla midpoint OR ADX < 20 (losing trend)
            if price > mid or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Volume_ADX_Filter"
timeframe = "4h"
leverage = 1.0