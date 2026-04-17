#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1d volume confirmation and 1w ADX trend filter.
Long when price breaks above Camarilla R1 AND volume > 1.3x 20-period average AND ADX > 20 (trending).
Short when price breaks below Camarilla S1 AND volume > 1.3x 20-period average AND ADX > 20.
Exit when price reverts to Camarilla midpoint ( (R1+S1)/2 ) OR ADX < 15 (range market).
Uses 12h for Camarilla calculation, 1d for volume, and 1w for ADX to reduce whipsaw and capture major trends.
Target: 50-150 total trades over 4 years (12-37/year). Camarilla levels provide precise intraday support/resistance,
volume confirms breakout strength, weekly ADX ensures we only trade in trending regimes, avoiding chop.
Works in bull markets (captures uptrends via R1 breakouts) and bear markets (captures downtrends via S1 breakdowns).
"""

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
    
    # Get 12h data for Camarilla calculation (using previous 12h bar's OHLC)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    open_12h = df_12h['open'].values
    
    # Calculate Camarilla levels for R1 and S1
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    camarilla_upper = close_12h + (high_12h - low_12h) * 1.1 / 12
    camarilla_lower = close_12h - (high_12h - low_12h) * 1.1 / 12
    camarilla_mid = (camarilla_upper + camarilla_lower) / 2  # midpoint for exit
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    # Get 1w data for ADX filter (weekly trend)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX on 1w timeframe (14-period)
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    close_1w_series = pd.Series(close_1w)
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Plus Directional Movement (+DM)
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smooth / np.where(atr != 0, atr, np.inf))
    minus_di = 100 * (minus_dm_smooth / np.where(atr != 0, atr, np.inf))
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), np.inf)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 12h timeframe
    camarilla_upper_aligned = align_htf_to_ltf(prices, df_12h, camarilla_upper)
    camarilla_lower_aligned = align_htf_to_ltf(prices, df_12h, camarilla_lower)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_12h, camarilla_mid)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_upper_aligned[i]) or np.isnan(camarilla_lower_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        cu = camarilla_upper_aligned[i]
        cl = camarilla_lower_aligned[i]
        cm = camarilla_mid_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Camarilla R1 AND volume > 1.3x avg AND ADX > 20 (trending)
            if price > cu and vol > 1.3 * vol_ma and adx_val > 20:
                signals[i] = 0.25
                position = 1
            # Short: price < Camarilla S1 AND volume > 1.3x avg AND ADX > 20 (trending)
            elif price < cl and vol > 1.3 * vol_ma and adx_val > 20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Camarilla midpoint OR ADX < 15 (range market)
            if price < cm or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Camarilla midpoint OR ADX < 15 (range market)
            if price > cm or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Volume_1wADX_Filter"
timeframe = "12h"
leverage = 1.0