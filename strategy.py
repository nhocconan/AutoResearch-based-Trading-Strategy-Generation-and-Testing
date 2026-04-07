#!/usr/bin/env python3
"""
4H Keltner Breakout with Volume Confirmation and 1D Trend Filter
Long when price breaks above upper Keltner channel (2*ATR) with expanding volume AND 1D EMA trend up
Short when price breaks below lower Keltner channel with expanding volume AND 1D EMA trend down
Exit when price crosses back to middle line
Uses ATR-based bands that adapt to volatility, reducing false breakouts in ranging markets.
Target: 20-30 trades per year per symbol (<120 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_keltner_breakout_volume_1d_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === ATR (14) for Keltner channels ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Keltner Channels (20-period EMA ± 2*ATR) ===
    ema_mid = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = ema_mid + 2 * atr
    keltner_lower = ema_mid - 2 * atr
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)  # Avoid division by zero
    
    # === 1D trend filter (EMA 21) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(ema_mid[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below middle line
            if close[i] < ema_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above middle line
            if close[i] > ema_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.2:
                signals[i] = 0.0
                continue
            
            # Entry: Keltner breakout with volume confirmation AND 1D trend filter
            if close[i] > keltner_upper[i] and ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                # Breakout above upper channel with rising 1D EMA -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < keltner_lower[i] and ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                # Breakdown below lower channel with falling 1D EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals