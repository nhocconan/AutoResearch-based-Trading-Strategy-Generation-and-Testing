#!/usr/bin/env python3
# 4h_Supertrend_VolumeBreakout
# Hypothesis: Supertrend (ATR=10, mult=3) identifies trend direction, while price breaking above/below ATR-based channels with volume confirmation captures institutional breakouts. Works in bull/bear: Supertrend avoids counter-trend trades, volume filter ensures momentum validity. ATR stops manage risk.

name = "4h_Supertrend_VolumeBreakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for Supertrend
    atr_period = 10
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = np.full_like(tr, np.nan)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.mean(tr[0:atr_period])
        for i in range(atr_period, len(tr)):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Supertrend calculation
    factor = 3.0
    hl2 = (high + low) / 2
    upper_band = hl2 + factor * atr
    lower_band = hl2 - factor * atr
    
    supertrend = np.full_like(close, np.nan)
    uptrend = np.full_like(close, True)
    
    for i in range(1, len(close)):
        if np.isnan(atr[i-1]) or np.isnan(upper_band[i-1]) or np.isnan(lower_band[i-1]):
            supertrend[i] = np.nan
            uptrend[i] = uptrend[i-1] if i > 0 else True
            continue
            
        if close[i] > upper_band[i-1]:
            uptrend[i] = True
        elif close[i] < lower_band[i-1]:
            uptrend[i] = False
        else:
            uptrend[i] = uptrend[i-1]
            if uptrend[i] and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if not uptrend[i] and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        supertrend[i] = lower_band[i] if uptrend[i] else upper_band[i]
    
    # Align Supertrend to 4h (already on 4h, but ensure alignment for HTF consistency)
    # No alignment needed as calculated on same timeframe
    
    # Calculate ATR-based channels for breakout detection
    atr_ma_period = 10
    atr_smooth = np.full_like(atr, np.nan)
    if len(atr) >= atr_ma_period:
        atr_smooth[atr_ma_period-1] = np.mean(atr[0:atr_ma_period])
        for i in range(atr_ma_period, len(atr)):
            atr_smooth[i] = (atr_smooth[i-1] * (atr_ma_period-1) + atr[i]) / atr_ma_period
    
    upper_channel = hl2 + 2.0 * atr_smooth
    lower_channel = hl2 - 2.0 * atr_smooth
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma_period = 20
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= vol_ma_period:
        vol_ma[vol_ma_period-1] = np.mean(volume[0:vol_ma_period])
        for i in range(vol_ma_period, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * (vol_ma_period-1) + volume[i]) / vol_ma_period
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period, atr_ma_period, vol_ma_period) + 5
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper channel AND uptrend (Supertrend long) AND volume spike
            if (close[i] > upper_channel[i] and 
                close[i] > supertrend[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower channel AND downtrend (Supertrend short) AND volume spike
            elif (close[i] < lower_channel[i] and 
                  close[i] < supertrend[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower channel OR trend reversal (Supertrend short)
            if close[i] < lower_channel[i] or close[i] < supertrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper channel OR trend reversal (Supertrend long)
            if close[i] > upper_channel[i] or close[i] > supertrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals