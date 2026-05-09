#!/usr/bin/env python3
# 12h_Donchian_Breakout_1wTrend_Volume
# Hypothesis: Uses weekly trend via Donchian breakout on 12h timeframe. Long when price breaks above 1w Donchian high with volume confirmation; short when breaks below 1w Donchian low. Uses 1d ATR for volatility filter to avoid chop. Designed for low-frequency, high-conviction trades in both bull and bear markets by following major trends. Target: 15-25 trades/year per symbol.

name = "12h_Donchian_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    donchian_high_1w = np.full_like(high_1w, np.nan)
    donchian_low_1w = np.full_like(low_1w, np.nan)
    
    if len(high_1w) >= 20:
        donchian_high_1w[19] = np.max(high_1w[0:20])
        donchian_low_1w[19] = np.min(low_1w[0:20])
        for i in range(20, len(high_1w)):
            donchian_high_1w[i] = max(donchian_high_1w[i-1], high_1w[i])
            donchian_low_1w[i] = min(donchian_low_1w[i-1], low_1w[i])
            # Remove oldest value from window
            if i >= 20:
                if high_1w[i-20] == donchian_high_1w[i-1]:
                    donchian_high_1w[i] = np.max(high_1w[i-19:i+1])
                if low_1w[i-20] == donchian_low_1w[i-1]:
                    donchian_low_1w[i] = np.min(low_1w[i-19:i+1])
    
    # Get daily ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ATR(14)
    tr = np.zeros_like(high_1d)
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]), 
                   abs(low_1d[i] - close_1d[i-1]))
    
    atr_1d = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_1d[13] = np.mean(tr[0:14])
        for i in range(14, len(tr)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align indicators to 12h timeframe
    donchian_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume filter: 12h volume / 20-period average volume
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
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(donchian_high_1w_aligned[i]) or np.isnan(donchian_low_1w_aligned[i]) or \
           np.isnan(atr_1d_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid extremely low volatility (choppy) markets
        # Skip if ATR is too low relative to price (avoid noise)
        if atr_1d_aligned[i] < 0.005 * close[i]:  # Less than 0.5% of price
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above weekly Donchian high with volume confirmation
            if close[i] > donchian_high_1w_aligned[i] and volume_ratio[i] > 1.8:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below weekly Donchian low with volume confirmation
            elif close[i] < donchian_low_1w_aligned[i] and volume_ratio[i] > 1.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price breaks below weekly Donchian low (trend reversal)
            if close[i] < donchian_low_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price breaks above weekly Donchian high (trend reversal)
            if close[i] > donchian_high_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals