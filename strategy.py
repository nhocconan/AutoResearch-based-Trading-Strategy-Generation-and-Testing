#!/usr/bin/env python3
# 1d_1w_cam_breakout_vol_v1
# Strategy: Daily timeframe using weekly Camarilla pivot breakouts with volume confirmation and ADX trend filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Weekly Camarilla levels (R4/S4) act as strong support/resistance. Breakouts with volume 
# and ADX>25 capture sustained moves in both bull and bear markets. Low-frequency signals reduce fee drag.
# Targets 20-60 trades over 4 years to minimize costs.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_cam_breakout_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly OHLC for Camarilla pivots
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r4_1w = close_1w + range_1w * 1.1 / 2.0
    r3_1w = close_1w + range_1w * 1.1 / 4.0
    s3_1w = close_1w - range_1w * 1.1 / 4.0
    s4_1w = close_1w - range_1w * 1.1 / 2.0
    
    # Weekly ADX for trend filter (14-period)
    # TR = max(H-L, abs(H-PC), abs(L-PC))
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # +DM and -DM
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    plus_dm14 = pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values
    minus_dm14 = pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values
    
    # DI and DX
    plus_di14 = 100 * plus_dm14 / tr14
    minus_di14 = 100 * minus_dm14 / tr14
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Align weekly data to daily timeframe (wait for weekly close)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)  # Strong volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(adx_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_1w_aligned[i] > 25
        
        # Weekly Camarilla breakout signals
        breakout_up = price_close > r4_1w_aligned[i]   # Break above weekly R4
        breakdown_down = price_close < s4_1w_aligned[i]  # Break below weekly S4
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: Break above weekly R4 with volume and strong trend
        long_signal = breakout_up and vol_confirmed and strong_trend
        
        # Short: Break below weekly S4 with volume and strong trend
        short_signal = breakdown_down and vol_confirmed and strong_trend
        
        # Exit when price returns to weekly pivot level or opposite S3/R3 level
        pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
        exit_long = position == 1 and (price_close < pivot_1w_aligned[i] or 
                                       price_close < s3_1w_aligned[i])
        exit_short = position == -1 and (price_close > pivot_1w_aligned[i] or 
                                         price_close > r3_1w_aligned[i])
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals