#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Trend_With_DMI"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range
    range_1d = high_1d - low_1d
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = close_1d + (range_1d * 1.1 / 2)
    camarilla_s3 = close_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Daily EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_1d = close_1d_series.ewm(span=34, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # DMI (ADX) on daily timeframe for trend strength
    # Calculate +DI, -DI, and ADX
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    
    # True Range
    tr1 = high_1d_series - low_1d_series
    tr2 = abs(high_1d_series - close_1d_series.shift(1))
    tr3 = abs(low_1d_series - close_1d_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high_1d_series.diff()
    down_move = low_1d_series.diff().multiply(-1)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM
    tr_smooth = pd.Series(tr).ewm(span=14, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, min_periods=14).mean().values
    
    # Calculate +DI, -DI, and ADX
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, min_periods=14).mean().values
    
    # Align DMI components to 4h
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_4h = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_4h = align_htf_to_ltf(prices, df_1d, minus_di)
    
    # Volume filter: current volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 150
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(adx_4h[i]) or 
            np.isnan(plus_di_4h[i]) or np.isnan(minus_di_4h[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND above daily EMA34 AND ADX > 25 AND +DI > -DI AND volume spike
            if (close[i] > r3_4h[i] and close[i] > ema_1d_aligned[i] and 
                adx_4h[i] > 25 and plus_di_4h[i] > minus_di_4h[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND below daily EMA34 AND ADX > 25 AND -DI > +DI AND volume spike
            elif (close[i] < s3_4h[i] and close[i] < ema_1d_aligned[i] and 
                  adx_4h[i] > 25 and minus_di_4h[i] > plus_di_4h[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S3 OR below daily EMA34 OR ADX < 20
            if (close[i] < s3_4h[i] or close[i] < ema_1d_aligned[i] or adx_4h[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above R3 OR above daily EMA34 OR ADX < 20
            if (close[i] > r3_4h[i] or close[i] > ema_1d_aligned[i] or adx_4h[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals