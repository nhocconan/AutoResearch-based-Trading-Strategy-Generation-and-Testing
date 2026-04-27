#!/usr/bin/env python3
"""
4h_1D_Camarilla_R3_S3_Breakout_VolumeSpike_v2
Hypothesis: Refined version with stricter volume filter (3x average) and longer-term trend filter (1-week EMA) to reduce trade frequency while maintaining edge. 
Targets 15-25 trades/year to minimize fee drag and improve generalization across BTC/ETH/SOL.
Works in both bull and bear markets by capturing strong momentum moves with volume confirmation and trend alignment.
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
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Camarilla: Range = (H-L), R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = np.zeros(len(close_1w))
    for i in range(len(close_1w)):
        if i < 50:
            ema_50_1w[i] = np.mean(close_1w[:i+1])
        else:
            ema_50_1w[i] = (close_1w[i] * 2 + ema_50_1w[i-1] * 49) / 51
    
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
    
    # Calculate volume average (50-period for more stable average)
    vol_ma_50 = np.full(n, np.nan)
    for i in range(50, n):
        vol_ma_50[i] = np.mean(volume[i-50:i])
    
    # Align indicators to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(50, 14)  # volume MA needs 50, ATR needs 14
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma_50[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_50[i] if vol_ma_50[i] > 0 else 0
        
        # Volume confirmation: > 3x average volume (stricter than before)
        volume_confirmation = vol_ratio > 3.0
        
        # Trend filter: price above/below weekly EMA50
        trend_up = price > ema_50_1w_aligned[i]
        trend_down = price < ema_50_1w_aligned[i]
        
        # ATR volatility filter: avoid low volatility periods
        vol_filter = atr[i] > np.mean(atr[max(0, i-50):i+1]) * 0.5 if i >= 50 else True
        
        if position == 0:
            # Long: break above daily R3 with volume, trend up, and volatility
            if volume_confirmation and trend_up and vol_filter and price > camarilla_r3_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below daily S3 with volume, trend down, and volatility
            elif volume_confirmation and trend_down and vol_filter and price < camarilla_s3_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to daily midpoint or trend changes
            daily_mid = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if price < daily_mid or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price returns to daily midpoint or trend changes
            daily_mid = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if price > daily_mid or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_1D_Camarilla_R3_S3_Breakout_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0