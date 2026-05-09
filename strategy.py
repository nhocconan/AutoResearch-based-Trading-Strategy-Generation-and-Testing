#!/usr/bin/env python3
# 6h_Keltner_Breakout_Volume_Strength
# Hypothesis: Breakout above/below Keltner Channel (ATR-based) with volume >2x 20-bar average and price strength (close > open).
# Uses Keltner Channel as dynamic support/resistance that adapts to volatility.
# In high volatility environments, breakouts are more significant.
# Volume filter ensures only high-conviction moves trigger entries.
# Designed for 10-25 trades/year on 6h timeframe.

name = "6h_Keltner_Breakout_Volume_Strength"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for Keltner Channel
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First period has no previous close
    
    atr = np.full_like(tr, np.nan)
    if len(tr) >= 10:
        atr[9] = np.mean(tr[0:10])
        for i in range(10, len(tr)):
            atr[i] = (atr[i-1] * 9 + tr[i]) / 10
    
    # Calculate EMA for Keltner Channel middle line
    ema = np.full_like(close, np.nan)
    if len(close) >= 20:
        ema[19] = np.mean(close[0:20])
        for i in range(20, len(close)):
            ema[i] = (close[i] * 2 + ema[i-1] * 18) / 20
    
    # Keltner Channel: Upper = EMA + 2*ATR, Lower = EMA - 2*ATR
    keltner_upper = ema + 2 * atr
    keltner_lower = ema - 2 * atr
    
    # Volume filter: 6h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    # Price strength: close > open (bullish candle)
    price_strength = close > prices['open'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or \
           np.isnan(volume_ratio[i]) or np.isnan(price_strength[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above Keltner Upper AND volume confirmation AND bullish candle
            if close[i] > keltner_upper[i] and volume_ratio[i] > 2.0 and price_strength[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below Keltner Lower AND volume confirmation AND bearish candle
            elif close[i] < keltner_lower[i] and volume_ratio[i] > 2.0 and not price_strength[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below Keltner Lower (reversal signal)
            if close[i] < keltner_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above Keltner Upper (reversal signal)
            if close[i] > keltner_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals