#!/usr/bin/env python3
# 4h_Keltner_Channel_Breakout_Trend_Filter
# Hypothesis: Keltner Channel breakouts with trend filter and volume confirmation. 
# Works in bull/bear: Trend filter (200 EMA) avoids counter-trend trades, volume confirms breakout strength.
# Keltner Channels adapt to volatility, providing dynamic support/resistance. 
# Uses 20 EMA for middle band and 2x ATR for bands, with trend filter from 50 EMA.

name = "4h_Keltner_Channel_Breakout_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR for Keltner Channels
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr[13] = np.mean(tr[0:14])
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20 EMA for Keltner middle band
    ema_20 = np.full_like(close, np.nan)
    if len(close) >= 20:
        ema_20[19] = np.mean(close[0:20])
        for i in range(20, len(close)):
            ema_20[i] = (ema_20[i-1] * 19 + close[i]) / 20
    
    # Keltner Channel bands
    keltner_upper = ema_20 + 2 * atr
    keltner_lower = ema_20 - 2 * atr
    
    # Trend filter: 50 EMA
    ema_50 = np.full_like(close, np.nan)
    if len(close) >= 50:
        ema_50[49] = np.mean(close[0:50])
        for i in range(50, len(close)):
            ema_50[i] = (ema_50[i-1] * 49 + close[i]) / 50
    
    # Volume confirmation: current volume / 20-period average
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
    
    start_idx = max(50, 20)  # Ensure EMA50 and other indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or
            np.isnan(ema_50[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper Keltner band AND uptrend (price > EMA50) AND volume spike
            if (close[i] > keltner_upper[i] and 
                close[i] > ema_50[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Keltner band AND downtrend (price < EMA50) AND volume spike
            elif (close[i] < keltner_lower[i] and 
                  close[i] < ema_50[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower Keltner band OR trend reversal (price < EMA50)
            if close[i] < keltner_lower[i] or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper Keltner band OR trend reversal (price > EMA50)
            if close[i] > keltner_upper[i] or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals