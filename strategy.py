#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation.
In bull market (12h close > 12h EMA50): long on 20-bar high breakout.
In bear market (12h close < 12h EMA50): short on 20-bar low breakout.
Volume must be above 20-period average to confirm breakout strength.
This combines price channel breakout with trend filter and volume confirmation.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    twelve_h_close = df_12h['close'].values
    twelve_h_ema = pd.Series(twelve_h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    twelve_h_ema_aligned = align_htf_to_ltf(prices, df_12h, twelve_h_ema)  # already shifted
    
    # === DONCHIAN CHANNEL (LTF) ===
    lookback = 20
    high_max = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    low_min = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        if np.isnan(twelve_h_ema_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 12h EMA
        bull_trend = close[i] > twelve_h_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend turns bearish
            if close[i] < low_min[i] or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend turns bullish
            if close[i] > high_max[i] or bull_trend:
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
                # In bull market: long on breakout above Donchian high
                if high[i] > high_max[i-1]:  # Use previous bar's high to avoid look-ahead
                    position = 1
                    signals[i] = 0.25
            else:
                # In bear market: short on breakdown below Donchian low
                if low[i] < low_min[i-1]:  # Use previous bar's low to avoid look-ahead
                    position = -1
                    signals[i] = -0.25
    
    return signals