#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h volume spike filter and 12h ADX trend filter
# Long when price breaks above 4h Camarilla R3 level AND 12h volume > 1.5x 20-period average AND 12h ADX > 25
# Short when price breaks below 4h Camarilla S3 level AND 12h volume > 1.5x 20-period average AND 12h ADX > 25
# Exit when price crosses 4h Camarilla pivot point (mean reversion)
# Uses 4h primary timeframe with 12h HTF for volume and trend confirmation
# Volume and trend filters reduce false breakouts and overtrading
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Camarilla_R3S3_Breakout_12hVolume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for volume and ADX filters
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h volume spike filter
    vol_12h = df_12h['volume'].values
    if len(vol_12h) >= 20:
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        volume_filter_12h = vol_12h > (1.5 * vol_ma_20)
    else:
        volume_filter_12h = np.zeros(len(df_12h), dtype=bool)
    
    # Calculate 12h ADX (14-period)
    if len(df_12h) >= 30:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        
        # True Range
        tr1 = high_12h[1:] - low_12h[1:]
        tr2 = np.abs(high_12h[1:] - close_12h[:-1])
        tr3 = np.abs(low_12h[1:] - close_12h[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        up_move = high_12h[1:] - high_12h[:-1]
        down_move = low_12h[:-1] - low_12h[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed values
        tr_period = 14
        atr = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values / atr
        
        # ADX
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
        
        adx_filter = adx > 25
    else:
        adx_filter = np.zeros(len(df_12h), dtype=bool)
    
    # Align 12h filters to 4h timeframe
    volume_filter_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_filter_12h)
    adx_filter_aligned = align_htf_to_ltf(prices, df_12h, adx_filter)
    
    # Get 4h data ONCE before loop for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    camarilla_r3 = close_4h + (1.1 * (high_4h - low_4h) / 2)
    camarilla_s3 = close_4h - (1.1 * (high_4h - low_4h) / 2)
    camarilla_pivot = (high_4h + low_4h + close_4h) / 3  # Standard pivot point
    
    # Align Camarilla levels to 4h timeframe (same df_4h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(volume_filter_12h_aligned[i]) or 
            np.isnan(adx_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND volume spike AND ADX > 25
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_filter_12h_aligned[i] and 
                adx_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND volume spike AND ADX > 25
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_filter_12h_aligned[i] and 
                  adx_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot (mean reversion)
            if close[i] < camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot (mean reversion)
            if close[i] > camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals