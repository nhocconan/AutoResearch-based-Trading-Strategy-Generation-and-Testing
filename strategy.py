#!/usr/bin/env python3
"""
12h Donchian Breakout with 1d Trend Filter and Volume Confirmation
Long when price breaks above Donchian upper band (20-period) with 1d uptrend and volume surge
Short when price breaks below Donchian lower band (20-period) with 1d downtrend and volume surge
Exit when price crosses the midline (10-period average of bands)
Uses volume confirmation to reduce false breakouts and works in both bull/bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Donchian Channels (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    # === 1d Trend Filter (using EMA) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=20, min_periods=20).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ratio[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below midline
            if close[i] < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above midline
            if close[i] > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume surge (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry conditions
            bullish_breakout = close[i] > highest_high[i]
            bearish_breakout = close[i] < lowest_low[i]
            uptrend = ema_1d_aligned[i] > ema_1d_aligned[i-1] if i > 0 else False
            downtrend = ema_1d_aligned[i] < ema_1d_aligned[i-1] if i > 0 else False
            
            if bullish_breakout and uptrend:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout and downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals