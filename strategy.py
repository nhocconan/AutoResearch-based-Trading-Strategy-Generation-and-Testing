#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1w volume spike and 1d ADX trend filter.
Long when price breaks above Camarilla R1 AND 1w volume > 1.5x average AND 1d ADX > 25.
Short when price breaks below Camarilla S1 AND 1w volume > 1.5x average AND 1d ADX > 25.
Exit when price reverts to Camarilla midpoint (PP) OR 1d ADX < 20.
Uses 12h for price action, 1w for volume confirmation to avoid fakeouts, 1d ADX for trend filter.
Target: 50-150 total trades over 4 years (12-37/year). Camarilla levels provide precise intraday support/resistance,
volume spike confirms institutional interest, daily ADX ensures we only trade in strong trends.
Works in bull markets (captures uptrends via R1 breakouts) and bear markets (captures downtrends via S1 breakdowns).
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
    
    # Get 12h data for Camarilla calculation (using previous day's OHLC)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels from previous 12h bar's OHLC
    # Camarilla formulas:
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    
    hl_range = high_12h - low_12h
    pp = (high_12h + low_12h + close_12h) / 3
    r1 = close_12h + (hl_range * 1.1 / 12)
    s1 = close_12h - (hl_range * 1.1 / 12)
    
    # Get 1w data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w volume average (20-period)
    volume_1w_series = pd.Series(volume_1w)
    volume_ma_1w = volume_1w_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    
    # Plus Directional Movement (+DM)
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
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
    
    # Align all indicators to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp)
    volume_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_1w)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(volume_ma_1w_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pp_val = pp_aligned[i]
        vol_ma = volume_ma_1w_aligned[i]
        adx_val = adx_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Camarilla R1 AND 1w volume > 1.5x avg AND 1d ADX > 25 (trending)
            if price > r1_val and vol > 1.5 * vol_ma and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price < Camarilla S1 AND 1w volume > 1.5x avg AND 1d ADX > 25 (trending)
            elif price < s1_val and vol > 1.5 * vol_ma and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Camarilla PP OR 1d ADX < 20 (range market)
            if price < pp_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Camarilla PP OR 1d ADX < 20 (range market)
            if price > pp_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_1wVolume_1dADX_Filter"
timeframe = "12h"
leverage = 1.0