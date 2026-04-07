#!/usr/bin/env python3
"""
12h_keltner_channel_1d_trend_volume_v2
Hypothesis: On 12h timeframe, use Keltner Channel breakouts for entry signals, filtered by 1d EMA trend and volume confirmation. 
In bull markets, price breaks above upper KC with volume; in bear markets, price breaks below lower KC with volume. 
1d EMA filter ensures alignment with higher timeframe trend, reducing whipsaw. Volume confirms genuine breakouts.
Target: 15-30 trades/year (~60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_keltner_channel_1d_trend_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter (more responsive than 200 for 12h trading)
    ema_50 = df_1d['close'].ewm(span=50, adjust=False).mean()
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
    
    # Keltner Channel (20, 2) on 12h: EMA20 ± 2*ATR(10)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False).mean()
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR
    atr = pd.Series(tr).ewm(span=10, adjust=False).mean()
    kc_upper = ema_20 + 2 * atr
    kc_lower = ema_20 - 2 * atr
    
    # Volume confirmation (20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below EMA20 or below EMA50 (trend filter)
            if close[i] < ema_20[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above EMA20 or above EMA50 (trend filter)
            if close[i] > ema_20[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper KC, with volume and price above EMA50
            if (close[i] > kc_upper[i] and vol_confirm and 
                close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower KC, with volume and price below EMA50
            elif (close[i] < kc_lower[i] and vol_confirm and 
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals