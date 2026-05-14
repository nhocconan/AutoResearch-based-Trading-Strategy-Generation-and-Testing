#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and ADX regime filter.
# Long when price breaks above Camarilla R3 (1d) AND 4h volume > 2.0x 20-period average AND ADX(14) > 25 (trending market).
# Short when price breaks below Camarilla S3 (1d) AND 4h volume > 2.0x 20-period average AND ADX(14) > 25.
# Exit on break of opposite Camarilla level (R2 for longs, S2 for shorts) or when ADX < 20 (range regime).
# Uses 1d HTF for Camarilla levels to reduce noise and overtrading vs shorter HTF.
# Volume confirmation reduces false breakouts. ADX filter ensures we only trade in trending regimes.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

name = "4h_Camarilla_R3S3_Breakout_1dVolumeSpike_ADXFilter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 4h Indicators (LTF) ---
    # 4h volume confirmation: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_4h = volume > (2.0 * vol_ma_20)
    
    # 4h ADX(14) for regime filter
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[open_[0]], open_[:-1]]))
    tr3 = np.abs(low - np.concatenate([[open_[0]], open_[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Directional Movement
    up_move = high - np.concatenate([[high[0]], high[:-1]])
    down_move = np.concatenate([[low[0]], low[:-1]]) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth DM and TR
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.25*(high-low), etc.
    # We use R3 and S3 for breakout, R2 and S2 for exit
    camarilla_range = high_1d - low_1d
    r3_1d = close_1d + 1.25 * camarilla_range
    s3_1d = close_1d - 1.25 * camarilla_range
    r2_1d = close_1d + 1.125 * camarilla_range  # exit for longs
    s2_1d = close_1d - 1.125 * camarilla_range  # exit for shorts
    
    # Align 1d indicators to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(r2_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or
            np.isnan(volume_confirm_4h[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 (1d) AND volume confirm AND ADX > 25 (trending)
            if (close[i] > r3_1d_aligned[i] and 
                volume_confirm_4h[i] and 
                adx[i] > 25):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 (1d) AND volume confirm AND ADX > 25 (trending)
            elif (close[i] < s3_1d_aligned[i] and 
                  volume_confirm_4h[i] and 
                  adx[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below R2 (1d) OR ADX < 20 (range regime)
            if (close[i] < r2_1d_aligned[i] or adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above S2 (1d) OR ADX < 20 (range regime)
            if (close[i] > s2_1d_aligned[i] or adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals