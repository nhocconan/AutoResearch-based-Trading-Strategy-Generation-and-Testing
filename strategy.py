#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike.
- Uses 1d HTF for trend alignment (proven in top performers)
- Camarilla H3/L3 from prior 1d for tighter structure than H4/L4
- Long: price breaks above H3 + volume > 2.0x 20-period avg + price > 1d EMA34
- Short: price breaks below L3 + volume > 2.0x 20-period avg + price < 1d EMA34
- Exit: price re-enters Camarilla H3-L3 range OR 1d EMA34 trend flip
- Discrete position sizing: ±0.25 to minimize fee churn
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
"""

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
    
    # Volume confirmation: > 2.0x 20-period average (tight to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA34 for trend filter (HTF = 1d as per top performers)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (based on prior 1d OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla formula: range = high - low
    # H3 = close + (high - low) * 1.1/4
    # L3 = close - (high - low) * 1.1/4
    rng = high_1d - low_1d
    camarilla_h3 = close_1d_prev + rng * (1.1 / 4)
    camarilla_l3 = close_1d_prev - rng * (1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 34, 20)  # Need 50 for safety, 34 for EMA34, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(h3_aligned[i]) or
            np.isnan(l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above H3 + volume confirmation + price > 1d EMA34
            if (close[i] > h3_aligned[i] and 
                volume_confirm and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 + volume confirmation + price < 1d EMA34
            elif (close[i] < l3_aligned[i] and 
                  volume_confirm and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters below L3 (mean reversion) OR price < 1d EMA34 (trend flip)
            if close[i] < l3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters above H3 (mean reversion) OR price > 1d EMA34 (trend flip)
            if close[i] > h3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0