#!/usr/bin/env python3
"""
12h_keltner_breakout_1d_trend_volume_v1
Hypothesis: Price breaks above/below Keltner Channel with volume confirmation and daily trend alignment.
Keltner Channel uses ATR(10) and EMA(20) to capture volatility-based breakouts.
Works in bull markets (breakouts above upper channel) and bear markets (breakdowns below lower channel).
Target: 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_keltner_breakout_1d_trend_volume_v1"
timeframe = "12h"
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
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Keltner Channel on 12h: EMA(20) ± ATR(10) * 2
    ema_20 = pd.Series(close).ewm(span=20, adjust=False).mean()
    atr_10 = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=10, min_periods=10).mean()
    upper_keltner = ema_20 + 2 * atr_10
    lower_keltner = ema_20 - 2 * atr_10
    
    # Daily EMA for trend filter (20-period)
    ema_20_1d = df_1d['close'].ewm(span=20, adjust=False).mean()
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d.values)
    
    # Volume confirmation (10-period average = 5 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_20[i]) or np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below EMA(20) or trend turns bearish
            if close[i] < ema_20[i] or close[i] < ema_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above EMA(20) or trend turns bullish
            if close[i] > ema_20[i] or close[i] > ema_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Keltner with volume and bullish trend
            if (close[i] > upper_keltner[i] and vol_confirm and 
                close[i] > ema_20_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Keltner with volume and bearish trend
            elif (close[i] < lower_keltner[i] and vol_confirm and 
                  close[i] < ema_20_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals