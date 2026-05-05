#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout + 1w ADX trend filter + volume confirmation
# Camarilla pivot levels identify key support/resistance from prior 1d range
# Long when: price breaks above R3 AND 1w ADX > 25 AND volume > 1.5x 20-period MA
# Short when: price breaks below S3 AND 1w ADX > 25 AND volume > 1.5x 20-period MA
# Exit when: price returns to Camarilla pivot point (PP) OR ADX drops below 20
# Uses Camarilla for structure, ADX for trend strength, volume for conviction
# Timeframe: 12h, HTF: 1w. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Camarilla_R3S3_1wADX_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from prior 1d (need 1d HTF data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1d OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: PP = (H+L+C)/3, range = H-L
    pp = (prev_high + prev_low + prev_close) / 3
    rng = prev_high - prev_low
    
    # Resistance levels: R3 = C + (H-L)*1.1/2, R4 = C + (H-L)*1.1
    # Support levels: S3 = C - (H-L)*1.1/2, S4 = C - (H-L)*1.1
    r3 = prev_close + rng * 1.1 / 2
    s3 = prev_close - rng * 1.1 / 2
    pp_level = pp  # pivot point for exit
    
    # Get 1w data ONCE before loop for ADX calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1w
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    if len(high_1w) >= 14:
        # True Range
        tr1 = np.abs(high_1w[1:] - low_1w[1:])
        tr2 = np.abs(high_1w[1:] - close_1w[:-1])
        tr3 = np.abs(low_1w[1:] - close_1w[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = high_1w[1:] - high_1w[:-1]
        down_move = low_1w[:-1] - low_1w[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed TR, +DM, -DM
        tr_period = 14
        atr = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False).mean().values
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False).mean().values
    else:
        adx = np.full(len(high_1w), np.nan)
    
    # ADX trend strength
    adx_strong = np.zeros(len(adx), dtype=bool)
    adx_weak = np.zeros(len(adx), dtype=bool)
    for i in range(len(adx)):
        if not np.isnan(adx[i]):
            adx_strong[i] = adx[i] > 25
            adx_weak[i] = adx[i] < 20
    
    # Align 1d Camarilla levels and 1w ADX to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_level)
    adx_strong_aligned = align_htf_to_ltf(prices, df_1w, adx_strong.astype(float))
    adx_weak_aligned = align_htf_to_ltf(prices, df_1w, adx_weak.astype(float))
    
    # Volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(adx_strong_aligned[i]) or np.isnan(adx_weak_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 + strong ADX + volume filter
            if (close[i] > r3_aligned[i] and 
                adx_strong_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 + strong ADX + volume filter
            elif (close[i] < s3_aligned[i] and 
                  adx_strong_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to pivot point OR weak ADX
            if (close[i] <= pp_aligned[i] or adx_weak_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to pivot point OR weak ADX
            if (close[i] >= pp_aligned[i] or adx_weak_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals