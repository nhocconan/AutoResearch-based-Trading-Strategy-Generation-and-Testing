#!/usr/bin/env python3
# 4h_Camarilla_Pullback_With_Volume
# Hypothesis: Buys pullbacks to Camarilla R3/S3 levels during strong 1d trends, with volume confirmation.
# Works in bull markets by buying dips in uptrends; works in bear markets by selling rallies in downtrends.
# Uses 1d trend filter to avoid counter-trend trades, volume to confirm institutional interest, and ATR stops for risk control.
# Target: 20-40 trades/year per symbol with disciplined risk management.

name = "4h_Camarilla_Pullback_With_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (ema_34_1d[i-1] * 33 + close_1d[i]) / 34
    
    # Calculate Camarilla levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3, S3 levels: R3 = close + 1.1*(high-low)/6, S3 = close - 1.1*(high-low)/6
    camarilla_r3_1d = close_1d + (1.1 * (high_1d - low_1d) / 6)
    camarilla_s3_1d = close_1d - (1.1 * (high_1d - low_1d) / 6)
    
    # Align Camarilla levels and EMA to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full_like(tr, np.nan)
    if len(tr) >= 10:
        atr[9] = np.mean(tr[0:10])
        for i in range(10, len(tr)):
            atr[i] = (atr[i-1] * 9 + tr[i]) / 10
    
    # Volume ratio: current vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend direction
        uptrend_1d = close[i] > ema_34_aligned[i]
        
        if position == 0:
            # Enter long: pullback to S3 in uptrend + volume confirmation
            if uptrend_1d and close[i] <= camarilla_s3_aligned[i] * 1.001 and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: pullback to R3 in downtrend + volume confirmation
            elif not uptrend_1d and close[i] >= camarilla_r3_aligned[i] * 0.999 and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal or ATR stop
            if not uptrend_1d or close[i] < camarilla_s3_aligned[i] or close[i] < close[i-1] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reversal or ATR stop
            if uptrend_1d or close[i] > camarilla_r3_aligned[i] or close[i] > close[i-1] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals