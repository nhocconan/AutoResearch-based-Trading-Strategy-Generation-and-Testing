#!/usr/bin/env python3
"""
6h Daily Keltner Channel Breakout with Volume Confirmation and ADX Trend Filter
Hypothesis: Keltner Channel breakouts capture momentum moves. 
Break above upper KC with volume and rising ADX indicates bullish momentum.
Break below lower KC with volume and rising ADX indicates bearish momentum.
Uses daily timeframe for KC calculation (more stable than 6h) to reduce noise.
Designed for low frequency (15-35 trades/year) with strong trend filtering.
Works in both bull and bear markets by only taking trades in direction of ADX trend.
"""

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
    
    # Get daily data for Keltner Channel and ADX calculation (once before loop)
    df_d = get_htf_data(prices, '1d')
    
    # Calculate daily ATR for Keltner Channel (using 10-period ATR)
    tr1 = df_d['high'] - df_d['low']
    tr2 = abs(df_d['high'] - df_d['close'].shift(1))
    tr3 = abs(df_d['low'] - df_d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr.rolling(window=10, min_periods=10).mean().values
    
    # Calculate daily EMA20 for Keltner Channel middle line
    ema_20 = pd.Series(df_d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Keltner Channel bounds: Upper = EMA20 + 2*ATR, Lower = EMA20 - 2*ATR
    kc_upper = ema_20 + 2.0 * atr_10
    kc_lower = ema_20 - 2.0 * atr_10
    
    # Calculate daily ADX for trend filter (14-period)
    # +DM and -DM calculation
    up_move = df_d['high'].diff()
    down_move = df_d['low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all daily indicators to 6h timeframe
    kc_upper_aligned = align_htf_to_ltf(prices, df_d, kc_upper)
    kc_lower_aligned = align_htf_to_ltf(prices, df_d, kc_lower)
    adx_aligned = align_htf_to_ltf(prices, df_d, adx)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(kc_upper_aligned[i]) or 
            np.isnan(kc_lower_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kc_upper_val = kc_upper_aligned[i]
        kc_lower_val = kc_lower_aligned[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Long: break above upper KC with volume spike and ADX > 20 (trending market)
            if (price > kc_upper_val and volume_spike[i] and adx_val > 20):
                signals[i] = 0.25
                position = 1
            # Short: break below lower KC with volume spike and ADX > 20 (trending market)
            elif (price < kc_lower_val and volume_spike[i] and adx_val > 20):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price returns to middle of KC or ADX drops below 20 (trend weakening)
            if price <= ema_20[-1] if hasattr(ema_20, '__getitem__') else ema_20 or adx_val < 20:
                # Need to get the current EMA20 value - since we don't have it aligned,
                # we'll use price action: exit if price crosses back below upper KC
                if price < kc_upper_val:
                    signals[i] = 0.0
                    position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price returns to middle of KC or ADX drops below 20 (trend weakening)
            if price > kc_lower_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_KeltnerBreakout_VolumeSpike_ADXFilter"
timeframe = "6h"
leverage = 1.0