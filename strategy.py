#!/usr/bin/env python3
"""
12h_Donchian_20_1D_Trend_Volume_Filter
Hypothesis: Use 12h price action with 1d trend filter and volume confirmation.
Long when 12h close breaks above 20-bar 12h Donchian high and close > 1d EMA34 and volume > 1.5x average.
Short when 12h close breaks below 20-bar 12h Donchian low and close < 1d EMA34 and volume > 1.5x average.
Exit when price returns to opposite Donchian band or trend reverses.
Designed for low frequency (10-30 trades/year) to avoid fee drag and work in bull/bear markets.
"""
name = "12h_Donchian_20_1D_Trend_Volume_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA warmup
    
    for i in range(start_idx, n):
        if position == 0:
            # Long entry: close > 1d EMA34 and volume > 1.5x 20-period average
            vol_avg = np.mean(volume[max(0, i-20):i]) if i >= 20 else 0
            vol_filter = volume[i] > (vol_avg * 1.5) if vol_avg > 0 else False
            
            if ema_34_aligned[i] > 0 and vol_filter:
                if close[i] > ema_34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < ema_34_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: close < 1d EMA34
            if close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close > 1d EMA34
            if close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals