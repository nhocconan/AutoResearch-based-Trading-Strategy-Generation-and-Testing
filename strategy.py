#!/usr/bin/env python3
"""
6h_RSI_Extremes_With_Volume_and_Trend
Hypothesis: RSI extremes (RSI<20 for long, RSI>80 for short) on 6h timeframe, filtered by 1d trend (price > EMA50 for long, price < EMA50 for short) and volume spike (>1.5x 24-period average). Works in both bull and bear markets by buying oversold dips in uptrends and selling overbought rallies in downtrends. Designed for low trade frequency (12-37/year) to minimize fee drag.
"""

name = "6h_RSI_Extremes_With_Volume_and_Trend"
timeframe = "6h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (ema_50_1d[i-1] * 49 + close_1d[i]) / 50
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI(14) on 6h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    rsi_gain = np.full_like(gain, np.nan)
    rsi_loss = np.full_like(loss, np.nan)
    if len(gain) >= 14:
        rsi_gain[13] = np.mean(gain[0:14])
        rsi_loss[13] = np.mean(loss[0:14])
        for i in range(14, len(gain)):
            rsi_gain[i] = (rsi_gain[i-1] * 13 + gain[i]) / 14
            rsi_loss[i] = (rsi_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full_like(close, np.nan)
    valid = (~np.isnan(rsi_gain)) & (~np.isnan(rsi_loss)) & (rsi_loss != 0)
    rs[valid] = rsi_gain[valid] / rsi_loss[valid]
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike filter: current volume / 24-period average volume (24*6h = 6 days)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 24:
        vol_ma[23] = np.mean(volume[0:24])
        for i in range(24, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 23 + volume[i]) / 24
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(24, 14)  # Ensure volume MA and RSI are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: RSI < 20 (oversold) AND uptrend (price > EMA50) AND volume spike
            if (rsi[i] < 20 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: RSI > 80 (overbought) AND downtrend (price < EMA50) AND volume spike
            elif (rsi[i] > 80 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = 0.25
            else:
                # Exit long: RSI > 60 (overbought) OR trend reversal (price < EMA50)
                if rsi[i] > 60 or close[i] < ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = -0.25
            else:
                # Exit short: RSI < 40 (oversold) OR trend reversal (price > EMA50)
                if rsi[i] < 40 or close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals