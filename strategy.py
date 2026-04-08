#!/usr/bin/env python3
"""
6H Weekly Trend + Daily Volume Filter
Hypothesis: Combining weekly trend direction with daily volume spikes on 6h timeframe captures
strong momentum moves while avoiding chop. Weekly trend provides directional bias,
daily volume filters for conviction, and 6s entry timing avoids false breaks.
Works in bull (rides trends) and bear (captures sharp moves) markets.
Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_trend_daily_volume_v1"
timeframe = "6h"
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
    
    # Weekly data for trend
    df_1w = get_htf_data(prices, '1w')
    
    # Daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly EMA(34) for trend
    ema_34w = df_1w['close'].ewm(span=34, adjust=False).mean().values
    ema_34w_aligned = align_htf_to_ltf(prices, df_1w, ema_34w)
    
    # Daily volume average (20-period)
    vol_ma_d = df_1d['volume'].rolling(window=20, min_periods=20).mean().values
    vol_ma_d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34w_aligned[i]) or np.isnan(vol_ma_d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: weekly trend turns bearish
            if close[i] < ema_34w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: weekly trend turns bullish
            if close[i] > ema_34w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume spike: current volume > 2x daily average
            volume_spike = volume[i] > (vol_ma_d_aligned[i] * 2.0)
            
            # Long: price above weekly EMA + volume spike
            if (close[i] > ema_34w_aligned[i] and volume_spike):
                position = 1
                signals[i] = 0.25
            # Short: price below weekly EMA + volume spike
            elif (close[i] < ema_34w_aligned[i] and volume_spike):
                position = -1
                signals[i] = -0.25
    
    return signals