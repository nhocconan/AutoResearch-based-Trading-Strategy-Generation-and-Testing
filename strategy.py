#!/usr/bin/env python3
# Hypothesis: 1h mean reversion within 4h Camarilla bands with volume confirmation and session filter (08-20 UTC).
# Long when price touches S3 with volume > 1.5x average and RSI(14) < 30 in ranging market (ADX < 25).
# Short when price touches R3 with volume > 1.5x average and RSI(14) > 70 in ranging market (ADX < 25).
# Uses 4h Camarilla for structure, 1h for entry timing, and 1d ADX for regime filter.
# Discrete position sizing 0.20 to limit fee churn. Target: 60-150 total trades over 4 years.

name = "1h_Camarilla_S3_R3_MeanReversion_VolumeRSI_ADX_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h Camarilla levels from previous day (6x 4h bars)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    lookback = 6  # 6 * 4h = 24h approx
    if len(close_4h) < lookback + 1:
        return np.zeros(n)
    
    # Rolling max/min/close for previous "day" on 4h
    high_prev_4h = pd.Series(high_4h).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    low_prev_4h = pd.Series(low_4h).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    close_prev_4h = pd.Series(close_4h).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    camarilla_range = high_prev_4h - low_prev_4h
    r3_4h = close_prev_4h + 1.1 * camarilla_range / 2
    s3_4h = close_prev_4h - 1.1 * camarilla_range / 2
    
    # Align 4h Camarilla levels to 1h timeframe
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # Calculate 1h volume average (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Get 1d ADX(14) for regime filter (ranging market: ADX < 25)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    if len(close_1d) < 14 + 1:
        return np.zeros(n)
    
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff()
    tr3 = pd.Series(close_1d).diff()
    tr = pd.concat([tr1.abs(), tr2.abs(), tr3.abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / np.where(atr != 0, atr, np.nan)
    minus_di = 100 * minus_dm_smooth / np.where(atr != 0, atr, np.nan)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), np.nan)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback + 20, 14)  # Ensure sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(rsi[i]) or np.isnan(adx_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price touches S3 with volume spike and RSI oversold in ranging market
            if (close[i] <= s3_4h_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i] and 
                rsi[i] < 30 and 
                adx_aligned[i] < 25):
                signals[i] = 0.20
                position = 1
            # SHORT: Price touches R3 with volume spike and RSI overbought in ranging market
            elif (close[i] >= r3_4h_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i] and 
                  rsi[i] > 70 and 
                  adx_aligned[i] < 25):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to midpoint or RSI > 50
            if close[i] >= (r3_4h_aligned[i] + s3_4h_aligned[i]) / 2 or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price reverts to midpoint or RSI < 50
            if close[i] <= (r3_4h_aligned[i] + s3_4h_aligned[i]) / 2 or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals