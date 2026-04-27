#!/usr/bin/env python3
# 1d_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Breakout above/below daily Camarilla R3/S3 with volume > 2x average and weekly EMA34 trend filter.
# Works in both bull and bear markets by capturing strong momentum moves with volume confirmation and trend alignment.
# Weekly EMA34 trend filter reduces false signals during choppy periods.
# Target: 10-20 trades/year to minimize fee dust while capturing significant moves.

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
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for EMA34 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: Range = (H-L), R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34 = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema_34[33] = np.mean(close_1w[:34])
        for i in range(34, len(close_1w)):
            ema_34[i] = (close_1w[i] * 2 + ema_34[i-1] * 32) / 34
    
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
    
    # Align daily Camarilla levels to daily timeframe (no shift needed as we're on 1d)
    camarilla_r3_aligned = camarilla_r3  # Already on daily timeframe
    camarilla_s3_aligned = camarilla_s3  # Already on daily timeframe
    
    # Align weekly EMA34 to daily timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(34, 20)  # EMA34 needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 2x average volume
        volume_confirmation = vol_ratio > 2.0
        
        # Trend filter: price above/below weekly EMA34
        uptrend = price > ema_34_aligned[i]
        downtrend = price < ema_34_aligned[i]
        
        # ATR volatility filter: avoid low volatility periods
        # Only trade when ATR is above 50% of its 50-period average
        if i >= 50:
            atr_avg = np.mean(atr[i-50:i+1])
            vol_filter = atr[i] > atr_avg * 0.5
        else:
            vol_filter = True  # No filter during warmup
        
        if position == 0:
            # Long: break above daily R3 with volume, uptrend, and volatility
            if volume_confirmation and uptrend and vol_filter and price > camarilla_r3_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below daily S3 with volume, downtrend, and volatility
            elif volume_confirmation and downtrend and vol_filter and price < camarilla_s3_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to daily midpoint or trend changes
            daily_mid = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if price < daily_mid or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price returns to daily midpoint or trend changes
            daily_mid = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if price > daily_mid or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0