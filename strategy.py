#!/usr/bin/env python3
# 1D_1D_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Dyn
# Hypothesis: Camarilla R3/S3 breakout with daily EMA34 trend filter and volume spike confirmation.
# Works in bull/bear: EMA34 trend filter avoids counter-trend trades, volume confirms breakout strength.
# Camarilla levels provide institutional-grade support/resistance that adapts to volatility.
# Focus on high-probability breakouts to minimize trades and avoid fee drag.
# Timeframe: 1d (primary), HTF: 1w (optional but not used here)

name = "1D_1D_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike_Dyn"
timeframe = "1d"
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
    
    # Use daily data for all calculations (same timeframe)
    if len(close) < 2:
        return np.zeros(n)
    
    # Previous day's values for Camarilla calculation
    ph = np.concatenate([[high[0]], high[:-1]])  # previous high
    pl = np.concatenate([[low[0]], low[:-1]])   # previous low
    pc = np.concatenate([[close[0]], close[:-1]]) # previous close
    
    # Calculate Camarilla levels (R3, S3 are the key breakout levels)
    rang = ph - pl
    r3 = pc + 1.1 * rang * 1.1666  # R3 = Close + 1.1 * (High-Low) * 1.1666
    s3 = pc - 1.1 * rang * 1.1666  # S3 = Close - 1.1 * (High-Low) * 1.1666
    
    # Calculate EMA34 for trend filter
    ema_34 = np.full_like(close, np.nan)
    if len(close) >= 34:
        ema_34[33] = np.mean(close[0:34])
        for i in range(34, len(close)):
            ema_34[i] = (ema_34[i-1] * 33 + close[i]) / 34
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema_34[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3 AND uptrend (price > EMA34) AND volume spike
            if (close[i] > r3[i] and 
                close[i] > ema_34[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 AND downtrend (price < EMA34) AND volume spike
            elif (close[i] < s3[i] and 
                  close[i] < ema_34[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 OR trend reversal (price < EMA34)
            if close[i] < s3[i] or close[i] < ema_34[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R3 OR trend reversal (price > EMA34)
            if close[i] > r3[i] or close[i] > ema_34[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals