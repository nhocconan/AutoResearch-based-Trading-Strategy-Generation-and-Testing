#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with daily trend filter and volume confirmation
# Works in bull markets via breakout momentum and in bear via mean-reversion at extreme levels
# Uses daily EMA34 for trend filter and volume spike for confirmation to reduce false signals
# Target: 20-50 trades/year to minimize fee drag while capturing significant moves

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily EMA34 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align daily EMA to 12h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate previous day's Camarilla levels (R3, S3)
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 20-period volume average for volume confirmation
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(34, vol_period) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Price breaks above Camarilla R3 with volume and trend confirmation
            if price > camarilla_r3_aligned[i] and vol_ratio > 2.0 and price > ema_1d_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Price breaks below Camarilla S3 with volume and trend confirmation
            elif price < camarilla_s3_aligned[i] and vol_ratio > 2.0 and price < ema_1d_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below Camarilla S3 or trend reversal
            if price < camarilla_s3_aligned[i] or price < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above Camarilla R3 or trend reversal
            if price > camarilla_r3_aligned[i] or price > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3S3_EMA34_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0