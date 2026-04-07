#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation.
Uses 12h EMA for trend direction (bull/bear filter) and 4h Donchian channels for entry.
In bull markets (price > 12h EMA): long on breakout above upper band.
In bear markets (price < 12h EMA): short on breakdown below lower band.
Volume must be above 20-period average to confirm breakout.
Exit on opposite Donchian band touch or trend reversal.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
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
    
    # === 12H TREND FILTER (HTF) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) == 0:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)  # already shifted
    
    # === 4H DONCHIAN CHANNELS (LTF) ===
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        if np.isnan(ema_12h_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 12h EMA
        bull_trend = close[i] > ema_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price touches lower Donchian band OR trend turns bearish
            if close[i] <= low_20[i] or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches upper Donchian band OR trend turns bullish
            if close[i] >= high_20[i] or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on 12h trend
            if bull_trend:
                # In bull market: long on breakout above upper band
                if close[i] > high_20[i]:
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear market: short on breakdown below lower band
                if close[i] < low_20[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals