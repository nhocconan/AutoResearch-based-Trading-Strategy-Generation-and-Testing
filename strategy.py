#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load weekly data for weekly pivot calculation (HTF)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using previous week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    pp_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pp_1w - low_1w
    s1_1w = 2 * pp_1w - high_1w
    r2_1w = pp_1w + (high_1w - low_1w)
    s2_1w = pp_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pp_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pp_1w)
    
    # Align weekly pivots to 6h timeframe
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Load daily data for volume confirmation (HTF)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period volume moving average on daily
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Load 6h data for price and volume
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate ATR for volatility filter (14-period on 6h)
    high_low = high_6h - low_6h
    high_close = np.abs(high_6h - np.roll(close_6h, 1))
    low_close = np.abs(low_6h - np.roll(close_6h, 1))
    high_low[0] = high_6h[0] - low_6h[0]
    high_close[0] = np.abs(high_6h[0] - close_6h[0])
    low_close[0] = np.abs(low_6h[0] - close_6h[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr_14_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_14_6h)
    
    # Calculate 20-period volume moving average on 6h
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if NaN in critical values
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(atr_14_6h_aligned[i]) or np.isnan(vol_ma_20_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol = volume_6h[i]
        
        # Volume conditions: current volume > 1.5x 6h MA AND > 1.5x daily MA
        vol_condition = (vol > 1.5 * vol_ma_20_6h_aligned[i]) and (vol > 1.5 * vol_ma_20_1d_aligned[i])
        
        if position == 0:
            # Long: price breaks above weekly R3 with volume confirmation
            if (price > r3_1w_aligned[i] and vol_condition):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S3 with volume confirmation
            elif (price < s3_1w_aligned[i] and vol_condition):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below weekly R2
            if price < r2_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above weekly S2
            if price > s2_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R3S3_Breakout_Volume"
timeframe = "6h"
leverage = 1.0