#!/usr/bin/env python3
# Strategy: 6h_1W_SR1_R3_S3_Fade
# Hypothesis: Price tends to fade from weekly S1/R1 and reverse from weekly S3/R3 levels.
# In both bull and bear markets, weekly support/resistance acts as key liquidity zones.
# Uses 1d timeframe for entry confirmation (price rejection at weekly levels) and volume.
# Targets 20-40 trades per year by requiring weekly level proximity and volume spike.
# Uses 6h as primary timeframe for lower noise and fewer false signals.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for SR levels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot and ranges
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Weekly S1, R1, S3, R3 levels (Camarilla-style)
    S1_1w = pivot_1w - (range_1w * 1.1 / 12)
    R1_1w = pivot_1w + (range_1w * 1.1 / 12)
    S3_1w = pivot_1w - (range_1w * 1.1 / 4)
    R3_1w = pivot_1w + (range_1w * 1.1 / 4)
    
    # Align weekly levels to 6h
    S1_1w_aligned = align_htf_to_ltf(prices, df_1w, S1_1w)
    R1_1w_aligned = align_htf_to_ltf(prices, df_1w, R1_1w)
    S3_1w_aligned = align_htf_to_ltf(prices, df_1w, S3_1w)
    R3_1w_aligned = align_htf_to_ltf(prices, df_1w, R3_1w)
    
    # Load daily data for trend and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily volume average for spike detection
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Load 6h data for entry timing and price action
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # 6h ATR for stop loss
    high_low = high_6h - low_6h
    high_close = np.abs(high_6h - np.roll(close_6h, 1))
    low_close = np.abs(low_6h - np.roll(close_6h, 1))
    high_low[0] = high_6h[0] - low_6h[0]
    high_close[0] = np.abs(high_6h[0] - close_6h[0])
    low_close[0] = np.abs(low_6h[0] - close_6h[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_6h, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(S1_1w_aligned[i]) or np.isnan(R1_1w_aligned[i]) or 
            np.isnan(S3_1w_aligned[i]) or np.isnan(R3_1w_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(atr_14_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        
        if position == 0:
            # Long setup: price near weekly S1 or S3 with bullish rejection
            near_S1 = abs(price - S1_1w_aligned[i]) < (0.5 * atr_14_aligned[i])
            near_S3 = abs(price - S3_1w_aligned[i]) < (0.5 * atr_14_aligned[i])
            bullish_rejection = (low_6h[i] <= min(S1_1w_aligned[i], S3_1w_aligned[i]) + 0.1 * atr_14_aligned[i]) and \
                               (close_6h[i] > open_6h[i]) if 'open_6h' in df_6h.columns else (close_6h[i] > low_6h[i])
            volume_confirm = vol > 1.5 * vol_ma_20_aligned[i]
            uptrend = price > ema34_1d_aligned[i]
            
            if ((near_S1 or near_S3) and bullish_rejection and volume_confirm and uptrend):
                signals[i] = 0.25
                position = 1
            
            # Short setup: price near weekly R1 or R3 with bearish rejection
            near_R1 = abs(price - R1_1w_aligned[i]) < (0.5 * atr_14_aligned[i])
            near_R3 = abs(price - R3_1w_aligned[i]) < (0.5 * atr_14_aligned[i])
            bearish_rejection = (high_6h[i] >= max(R1_1w_aligned[i], R3_1w_aligned[i]) - 0.1 * atr_14_aligned[i]) and \
                               (close_6h[i] < open_6h[i]) if 'open_6h' in df_6h.columns else (close_6h[i] < high_6h[i])
            downtrend = price < ema34_1d_aligned[i]
            
            if ((near_R1 or near_R3) and bearish_rejection and volume_confirm and downtrend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly S1 or ATR stop
            if (price < S1_1w_aligned[i] - 0.5 * atr_14_aligned[i] or 
                price < low_6h[i] - 2.0 * atr_14_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly R1 or ATR stop
            if (price > R1_1w_aligned[i] + 0.5 * atr_14_aligned[i] or 
                price > high_6h[i] + 2.0 * atr_14_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1W_SR1_R3_S3_Fade"
timeframe = "6h"
leverage = 1.0