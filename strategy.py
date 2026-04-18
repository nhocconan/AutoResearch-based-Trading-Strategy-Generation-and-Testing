#!/usr/bin/env python3
"""
12h Weekly EMA Trend with Daily Volume Spike
Hypothesis: On 12h timeframe, price retracements to the weekly EMA34 during strong volume spikes
continue in the direction of the weekly trend. Weekly EMA provides strong dynamic support/resistance,
while volume spikes confirm institutional interest. Works in both bull and bear markets by
only taking trades aligned with the weekly trend, avoiding counter-trend whipsaws.
Designed for 12-30 trades/year on 12h timeframe with tight entry conditions to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA (once before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34
    ema_34_w = pd.Series(df_w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_w_aligned = align_htf_to_ltf(prices, df_w, ema_34_w)
    
    # Get daily data for volume average
    df_d = get_htf_data(prices, '1d')
    vol_d = df_d['volume'].values
    # Daily 20-period volume average
    vol_ma_d = pd.Series(vol_d).rolling(window=20, min_periods=20).mean().values
    vol_ma_d_aligned = align_htf_to_ltf(prices, df_d, vol_ma_d)
    
    # ATR for stop loss (12h ATR)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_w_aligned[i]) or 
            np.isnan(vol_ma_d_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_w = ema_w_aligned[i]
        vol_ma = vol_ma_d_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price near weekly EMA (support) with volume spike in uptrend
            if price >= ema_w * 0.995 and price <= ema_w * 1.005 and volume[i] > 2.0 * vol_ma and price > ema_w:
                signals[i] = 0.25
                position = 1
            # Short: price near weekly EMA (resistance) with volume spike in downtrend
            elif price >= ema_w * 0.995 and price <= ema_w * 1.005 and volume[i] > 2.0 * vol_ma and price < ema_w:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price moves 2*ATR away from entry or reverses below EMA
            if price < ema_w or price < (high[i] - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price moves 2*ATR away from entry or reverses above EMA
            if price > ema_w or price > (low[i] + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_WeeklyEMA34_VolumeSpike_TrendFollow"
timeframe = "12h"
leverage = 1.0