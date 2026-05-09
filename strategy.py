#!/usr/bin/env python3
# 4h_RelativeStrength_Index_Extremes_VolumeConfirmation
# Hypothesis: Enter long when RSI(14) < 30 with volume > 1.8x 20-bar average, short when RSI > 70 with volume confirmation.
# Uses RSI mean reversion in ranging markets and momentum in trending markets. Volume filter ensures conviction.
# Designed for 15-25 trades/year on 4h timeframe to minimize fee drag while capturing reversals and continuations.

name = "4h_RelativeStrength_Index_Extremes_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    # Initialize first average
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[0:14])
        avg_loss[13] = np.mean(loss[0:14])
        
        # Wilder's smoothing
        for i in range(14, len(close)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full_like(close, np.nan)
    rsi = np.full_like(close, np.nan)
    valid = (avg_loss != 0) & ~np.isnan(avg_loss)
    rs[valid] = avg_gain[valid] / avg_loss[valid]
    rsi[valid] = 100 - (100 / (1 + rs[valid]))
    rsi[avg_loss == 0] = 100  # No loss = RSI 100
    
    # Volume filter: 20-period average
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
    
    start_idx = 20  # Need volume MA and RSI ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(rsi[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI < 30 (oversold) with volume confirmation
            if rsi[i] < 30 and volume_ratio[i] > 1.8:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI > 70 (overbought) with volume confirmation
            elif rsi[i] > 70 and volume_ratio[i] > 1.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI > 50 (mean reversion) or RSI > 70 (overbought)
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI < 50 (mean reversion) or RSI < 30 (oversold)
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals