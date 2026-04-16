#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R for mean reversion in ranging markets combined with 4h volume confirmation and ADX trend filter.
# Long when Williams %R < -80 (oversold), volume > 1.5x 20-period average, and ADX < 25 (ranging market).
# Short when Williams %R > -20 (overbought), volume > 1.5x 20-period average, and ADX < 25.
# Exit when Williams %R crosses above -50 for longs or below -50 for shorts.
# Uses discrete position size 0.25. Williams %R identifies extremes, volume confirms interest,
# ADX filter avoids false signals in strong trends. Target: 75-200 total trades over 4 years (19-50/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 13 or highest_high[i] == lowest_low[i]:
            williams_r[i] = -50.0  # neutral
        else:
            williams_r[i] = (highest_high[i] - close_1d[i]) / (highest_high[i] - lowest_low[i]) * -100
    
    # Align 1d Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 4h data once before loop for volume and ADX
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Volume moving average (20-period) on 4h
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # ADX calculation (14-period) on 4h
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_4h - np.roll(high_4h, 1)
    down_move = np.roll(low_4h, 1) - low_4h
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = np.zeros_like(close_4h)
    for i in range(len(close_4h)):
        if plus_di[i] + minus_di[i] == 0:
            dx[i] = 0
        else:
            dx[i] = np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100
    
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        wr = williams_r_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        adx_val = adx_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R crosses above -50
            if wr > -50:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R crosses below -50
            if wr < -50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 1.5x 20-period average (4h)
            vol_filter = vol > 1.5 * vol_ma_val
            
            # Trend filter: ADX < 25 (ranging market)
            trend_filter = adx_val < 25
            
            # LONG: Williams %R oversold (< -80), volume spike, ranging market
            if (wr < -80) and vol_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R overbought (> -20), volume spike, ranging market
            elif (wr > -20) and vol_filter and trend_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_1dWilliamsR_VolumeSpike_ADXFilter_V1"
timeframe = "4h"
leverage = 1.0