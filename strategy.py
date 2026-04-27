#!/usr/bin/env python3
"""
12h_1W_1D_Camarilla_R3_S3_Breakout_VolumeSpike
Hypothesis: Breakout above/below weekly and daily combined Camarilla R3/S3 levels with volume > 1.8x average and ATR volatility filter.
Uses weekly trend filter to avoid counter-trend moves. Designed for 12h timeframe to capture multi-day momentum moves.
Targets 15-25 trades/year to minimize fee decay while capturing significant moves in both bull and bear markets.
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
    
    # Get weekly and daily data for multi-timeframe confirmation
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[:34])
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = (close_1w[i] * 2/35) + (ema_34_1w[i-1] * 33/35)
    
    # Calculate daily Camarilla levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3_1d = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s3_1d = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Calculate ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros(n)
    for i in range(n):
        if i < 14:
            atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[0]
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    # Align weekly EMA34 and daily Camarilla levels to 12h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 34)  # volume MA needs 20, weekly EMA needs 34
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(camarilla_r3_1d_aligned[i]) or
            np.isnan(camarilla_s3_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 1.8x average volume (less strict than 2x to allow more trades)
        volume_confirmation = vol_ratio > 1.8
        
        # ATR volatility filter: avoid low volatility periods
        vol_filter = atr[i] > np.mean(atr[max(0, i-50):i+1]) * 0.4 if i >= 50 else True
        
        # Weekly trend filter: only take longs in uptrend, shorts in downtrend
        weekly_uptrend = price > ema_34_1w_aligned[i]
        weekly_downtrend = price < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: break above daily R3 with volume, volatility, and weekly uptrend
            if volume_confirmation and vol_filter and weekly_uptrend and price > camarilla_r3_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below daily S3 with volume, volatility, and weekly downtrend
            elif volume_confirmation and vol_filter and weekly_downtrend and price < camarilla_s3_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to daily midpoint or volatility drops significantly
            daily_mid = (camarilla_r3_1d_aligned[i] + camarilla_s3_1d_aligned[i]) / 2
            if price < daily_mid or atr[i] < np.mean(atr[max(0, i-50):i+1]) * 0.25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price returns to daily midpoint or volatility drops significantly
            daily_mid = (camarilla_r3_1d_aligned[i] + camarilla_s3_1d_aligned[i]) / 2
            if price > daily_mid or atr[i] < np.mean(atr[max(0, i-50):i+1]) * 0.25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "12h_1W_1D_Camarilla_R3_S3_Breakout_VolumeSpike"
timeframe = "12h"
leverage = 1.0