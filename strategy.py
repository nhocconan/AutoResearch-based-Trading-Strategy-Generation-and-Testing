#!/usr/bin/env python3
# 12h_Keltner_Breakout_VolumeSpike_Trend
# Hypothesis: Keltner breakout with volume spike and trend filter on 12h timeframe. Works in bull/bear by avoiding counter-trend trades.
# Uses Keltner channel (20-period, 2xATR) breakouts, volume > 1.5x average, and trend filter (price above/below EMA50).
# Keltner channels adapt to volatility, providing robust breakout signals in both trending and ranging markets.

name = "12h_Keltner_Breakout_VolumeSpike_Trend"
timeframe = "12h"
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
    
    # Calculate ATR for Keltner channels
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = np.full_like(close, np.nan)
    if len(tr) >= 20:
        atr[19] = np.mean(tr[0:20])
        for i in range(20, len(tr)):
            atr[i] = (atr[i-1] * 19 + tr[i]) / 20
    
    # Calculate Keltner channels: EMA20 ± 2*ATR
    ema20 = np.full_like(close, np.nan)
    if len(close) >= 20:
        ema20[19] = np.mean(close[0:20])
        for i in range(20, len(close)):
            ema20[i] = (ema20[i-1] * 19 + close[i]) / 20
    
    upper_keltner = ema20 + 2 * atr
    lower_keltner = ema20 - 2 * atr
    
    # Calculate EMA50 for trend filter
    ema50 = np.full_like(close, np.nan)
    if len(close) >= 50:
        ema50[49] = np.mean(close[0:50])
        for i in range(50, len(close)):
            ema50[i] = (ema50[i-1] * 49 + close[i]) / 50
    
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
    
    start_idx = max(20, 50)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or
            np.isnan(ema50[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper Keltner AND uptrend (price > EMA50) AND volume spike
            if (close[i] > upper_keltner[i] and 
                close[i] > ema50[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Keltner AND downtrend (price < EMA50) AND volume spike
            elif (close[i] < lower_keltner[i] and 
                  close[i] < ema50[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower Keltner OR trend reversal (price < EMA50)
            if close[i] < lower_keltner[i] or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper Keltner OR trend reversal (price > EMA50)
            if close[i] > upper_keltner[i] or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals