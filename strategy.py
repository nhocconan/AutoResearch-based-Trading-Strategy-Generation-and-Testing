#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike.
Long when price breaks above Camarilla R3 AND price > 1d EMA34 AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S3 AND price < 1d EMA34 AND volume > 2.0x 20-period average.
Exit when price reverts to Camarilla H5/L5 midpoint OR ATR trailing stop (2.5*ATR from extreme).
Uses 1d HTF for trend alignment and 1w for regime filter (ADX < 20 = choppy, avoid breakouts).
Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25.
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1w ADX for regime filter (avoid breakouts in choppy markets)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for ADX
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr_1w = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smoothed = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smoothed = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smoothed / atr_1w
    di_minus = 100 * dm_minus_smoothed / atr_1w
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1w = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate Camarilla levels from previous day to avoid look-ahead
    # Roll by 1 to use previous day's OHLC
    high_roll = np.roll(high, 1)
    low_roll = np.roll(low, 1)
    close_roll = np.roll(close, 1)
    high_roll[0] = np.nan
    low_roll[0] = np.nan
    close_roll[0] = np.nan
    
    # Camarilla levels: based on previous day's range
    rng = high_roll - low_roll
    camarilla_h5 = close_roll + 1.1 * rng / 6  # High 5
    camarilla_l5 = close_roll - 1.1 * rng / 6  # Low 5
    camarilla_h3 = close_roll + 1.1 * rng / 4  # High 3
    camarilla_s3 = close_roll - 1.1 * rng / 4  # Low 3
    camarilla_mid = (camarilla_h5 + camarilla_l5) / 2.0  # Midpoint of H5/L5
    
    # 4h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(34, 30, 1)  # ema_34_1d, adx_1w, and +1 for roll
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(adx_1w_aligned[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(camarilla_mid[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_1d_aligned[i]
        adx_val = adx_1w_aligned[i]
        h3 = camarilla_h3[i]
        s3 = camarilla_s3[i]
        mid = camarilla_mid[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        # Regime filter: only trade breakouts when ADX > 25 (trending market)
        in_trend = adx_val > 25
        
        if position == 0:
            # Long: price breaks above H3 AND price > 1d EMA34 AND volume spike AND trending market
            if price > h3 and price > ema_val and volume[i] > 2.0 * vol_ma_val and in_trend:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: price breaks below S3 AND price < 1d EMA34 AND volume spike AND trending market
            elif price < s3 and price < ema_val and volume[i] > 2.0 * vol_ma_val and in_trend:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price reverts to H5/L5 midpoint
            if position == 1 and price < mid:
                exit_signal = True
            elif position == -1 and price > mid:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_H3S3_Breakout_1dEMA34_Trend_VolumeSpike_H5L5Exit_ADXRegime_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0