#!/usr/bin/env python3
# 4h_Bollinger_Breakout_Volume_Trend
# Hypothesis: Uses Bollinger Band breakout with volume confirmation and 12h EMA trend filter.
# Designed to capture high-probability breakouts during volatility expansion in the direction of the higher timeframe trend.
# Works in bull and bear markets by filtering breakouts with 12h EMA to avoid counter-trend trades.
# Target: 20-35 trades/year per symbol with disciplined risk control.

name = "4h_Bollinger_Breakout_Volume_Trend"
timeframe = "4h"
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50) for trend filter
    ema_50_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[0:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (close_12h[i] * 2 + ema_50_12h[i-1] * 49) / 50
    
    # Align 12h EMA to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Bollinger Bands (20, 2) on 4h close
    sma_20 = np.full_like(close, np.nan)
    std_20 = np.full_like(close, np.nan)
    
    if len(close) >= 20:
        sma_20[19] = np.mean(close[0:20])
        std_20[19] = np.std(close[0:20])
        for i in range(20, len(close)):
            sma_20[i] = (sma_20[i-1] * 19 + close[i]) / 20
            std_20[i] = np.sqrt((std_20[i-1]**2 * 19 + (close[i] - sma_20[i])**2) / 20)
    
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Volume filter: 4h volume / 20-period average volume
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
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above upper BB AND volume confirmation AND price above 12h EMA50
            if close[i] > upper_bb[i] and volume_ratio[i] > 2.0 and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below lower BB AND volume confirmation AND price below 12h EMA50
            elif close[i] < lower_bb[i] and volume_ratio[i] > 2.0 and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price closes below middle Bollinger Band (SMA20)
            if close[i] < sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price closes above middle Bollinger Band (SMA20)
            if close[i] > sma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals