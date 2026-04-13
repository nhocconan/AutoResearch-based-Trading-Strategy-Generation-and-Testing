#!/usr/bin/env python3
"""
12h_1d_Volume_Spike_Pullback
Hypothesis: In bull or bear markets, strong volume spikes often precede short-term pullbacks.
Enter on pullback after volume spike with trend alignment from 1d EMA. Works in both regimes:
- Bull: buy pullbacks in uptrend
- Bear: sell rallies in downtrend
Target: 15-30 trades/year on 12h (60-120 total over 4 years).
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
    
    # Get 1d data for EMA trend and volume baseline
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(20) for trend
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1d volume average for spike detection (20-period)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 12h volume spike detection (volume > 2x 20-period average)
    vol_ma_20_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20_12h * 2.0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(vol_ma_20_12h[i]):
            signals[i] = 0.0
            continue
        
        # Trend condition: price relative to 1d EMA
        above_trend = close[i] > ema_20_1d_aligned[i]
        below_trend = close[i] < ema_20_1d_aligned[i]
        
        # Entry conditions
        if volume_spike[i]:
            # After volume spike, look for pullback
            if above_trend and close[i] < close[i-1]:  # Pullback in uptrend -> long
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
            elif below_trend and close[i] > close[i-1]:  # Rally in downtrend -> short
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
            else:
                # Hold current position
                if position == 1:
                    signals[i] = position_size
                elif position == -1:
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
        else:
            # No volume spike - maintain or exit
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_Volume_Spike_Pullback"
timeframe = "12h"
leverage = 1.0