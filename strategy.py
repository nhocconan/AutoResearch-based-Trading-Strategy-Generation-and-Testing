#!/usr/bin/env python3
"""
6H Keltner Breakout with Volume Confirmation and 12h Trend Filter
Long when price breaks above upper Keltner channel (2*ATR) with expanding volume AND 12h EMA trend up
Short when price breaks below lower Keltner channel with expanding volume AND 12h EMA trend down
Exit when price crosses back to middle line
Uses ATR-based bands that adapt to volatility, reducing false breakouts in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_keltner_breakout_volume_12h_trend_v1"
timeframe = "6h"
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
    
    # === 12h trend filter (EMA 21) ===
    df_12h = get_htf_data(prices, '12h')
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(ema_mid[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema_12h_aligned[i])):
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
            
            # Entry: Keltner breakout with volume confirmation AND 12h trend filter
            if close[i] > keltner_upper[i] and ema_12h_aligned[i] > ema_12h_aligned[i-1]:
                # Breakout above upper channel with rising 12h EMA -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < keltner_lower[i] and ema_12h_aligned[i] < ema_12h_aligned[i-1]:
                # Breakdown below lower channel with falling 12h EMA -> short
                position = -1
                signals[i] = -0.25
    
    return signals