#!/usr/bin/env python3
# 4h_Keltner_Breakout_VolumeSpike_EMATrend
# Hypothesis: Keltner channel breakouts with EMA trend filter and volume spike confirmation.
# Works in bull/bear: EMA trend filter avoids counter-trend trades, volume spike confirms institutional interest.
# Keltner channels adapt to volatility, providing dynamic support/resistance for breakouts.
# Uses EMA20 for trend and volume ratio (current/20-bar average) for confirmation.
# Target: 20-40 trades/year per symbol to minimize fee drag.

name = "4h_Keltner_Breakout_VolumeSpike_EMATrend"
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
    
    # Calculate Keltner Channel from 1h data (using 1h EMA and ATR)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    close_1h = df_1h['close'].values
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    
    # EMA20 for middle line
    ema_20 = np.full_like(close_1h, np.nan)
    if len(close_1h) >= 20:
        ema_20[19] = np.mean(close_1h[0:20])
        for i in range(20, len(close_1h)):
            ema_20[i] = (ema_20[i-1] * 19 + close_1h[i]) / 20
    
    # ATR(10) for channel width
    tr1 = high_1h[1:] - low_1h[1:]
    tr2 = np.abs(high_1h[1:] - close_1h[:-1])
    tr3 = np.abs(low_1h[1:] - close_1h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[0], tr])  # first TR is 0
    
    atr_10 = np.full_like(tr, np.nan)
    if len(tr) >= 10:
        atr_10[9] = np.mean(tr[0:10])
        for i in range(10, len(tr)):
            atr_10[i] = (atr_10[i-1] * 9 + tr[i]) / 10
    
    # Keltner Bands: EMA20 ± 2 * ATR(10)
    keltner_middle = ema_20
    keltner_upper = keltner_middle + 2 * atr_10
    keltner_lower = keltner_middle - 2 * atr_10
    
    # Align Keltner levels to 4h timeframe
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1h, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1h, keltner_lower)
    keltner_middle_aligned = align_htf_to_ltf(prices, df_1h, keltner_middle)
    
    # EMA50 trend filter from 4h data
    ema_50 = np.full_like(close, np.nan)
    if len(close) >= 50:
        ema_50[49] = np.mean(close[0:50])
        for i in range(50, len(close)):
            ema_50[i] = (ema_50[i-1] * 49 + close[i]) / 50
    
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
    
    start_idx = max(50, 20)  # Ensure EMA50 and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or
            np.isnan(ema_50[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper Keltner AND uptrend (price > EMA50) AND volume spike
            if (close[i] > keltner_upper_aligned[i] and 
                close[i] > ema_50[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Keltner AND downtrend (price < EMA50) AND volume spike
            elif (close[i] < keltner_lower_aligned[i] and 
                  close[i] < ema_50[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below middle Keltner OR trend reversal (price < EMA50)
            if close[i] < keltner_middle_aligned[i] or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above middle Keltner OR trend reversal (price > EMA50)
            if close[i] > keltner_middle_aligned[i] or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals